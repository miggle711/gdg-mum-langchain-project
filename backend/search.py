import logging
import time
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from app.config import settings
from app.metrics import search_cache_hits, search_cache_misses, rerank_latency, es_search_latency

logger = logging.getLogger(__name__)

ELASTICSEARCH_URL = settings.elasticsearch_url
ES_INDEX = "products"
REVIEWS_ES_INDEX = "reviews"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 768

_es: Optional[Elasticsearch] = None
_reranker = None
_embedding_model = None


def get_es() -> Elasticsearch:
    global _es
    if _es is None:
        _es = Elasticsearch(ELASTICSEARCH_URL)
    return _es


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder reranker: %s", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker loaded.")
    return _reranker


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embedding_model


def init_es_index() -> None:
    es = get_es()
    if es.indices.exists(index=ES_INDEX):
        logger.info("Elasticsearch index '%s' already exists, skipping creation.", ES_INDEX)
        return
    logger.info("Creating Elasticsearch index '%s'...", ES_INDEX)
    es.indices.create(index=ES_INDEX, body={
        "mappings": {
            "properties": {
                "id":             {"type": "keyword"},
                "name":           {"type": "text", "analyzer": "english"},
                "description":    {"type": "text", "analyzer": "english"},
                "category":       {"type": "keyword"},
                "price":          {"type": "float"},
                "original_price": {"type": "float"},
                "rating":         {"type": "float"},
                "reviews":        {"type": "integer"},
                "image":          {"type": "keyword", "index": False},
                "embedding":      {"type": "dense_vector", "dims": EMBEDDING_DIM, "index": True, "similarity": "cosine"},
            }
        }
    })


def init_reviews_index() -> None:
    es = get_es()
    if es.indices.exists(index=REVIEWS_ES_INDEX):
        logger.info("Elasticsearch index '%s' already exists, skipping creation.", REVIEWS_ES_INDEX)
        return
    logger.info("Creating Elasticsearch index '%s'...", REVIEWS_ES_INDEX)
    es.indices.create(index=REVIEWS_ES_INDEX, body={
        "mappings": {
            "properties": {
                "id":                {"type": "keyword"},
                "product_id":        {"type": "keyword"},
                "product_name":      {"type": "text", "analyzer": "english"},
                "title":             {"type": "text", "analyzer": "english"},
                "text":              {"type": "text", "analyzer": "english"},
                "rating":            {"type": "float"},
                "verified_purchase": {"type": "boolean"},
                "helpful_vote":      {"type": "integer"},
                "embedding":         {"type": "dense_vector", "dims": EMBEDDING_DIM, "index": True, "similarity": "cosine"},
            }
        }
    })


def es_bulk_index(documents: List[Dict[str, Any]]) -> None:
    from elasticsearch.helpers import bulk
    es = get_es()
    logger.info("Bulk indexing %d documents into ES index '%s'...", len(documents), ES_INDEX)

    def _actions():
        for doc in documents:
            yield {"_index": ES_INDEX, "_id": doc["id"], "_source": doc}

    bulk(es, _actions())
    logger.info("Bulk indexing complete.")


def es_bulk_index_reviews(documents: List[Dict[str, Any]]) -> None:
    from elasticsearch.helpers import bulk
    es = get_es()
    logger.info("Bulk indexing %d documents into ES index '%s'...", len(documents), REVIEWS_ES_INDEX)

    def _actions():
        for doc in documents:
            yield {"_index": REVIEWS_ES_INDEX, "_id": doc["id"], "_source": doc}

    bulk(es, _actions())
    logger.info("Bulk indexing complete.")


def es_upsert_document(doc: Dict[str, Any]) -> None:
    """Index (create or fully replace) a single product document. Used by
    the product write routes to keep ES in sync with Postgres without a
    CDC pipeline called in the same request as the Postgres
    write, after the Postgres commit succeeds."""
    es = get_es()
    es.index(index=ES_INDEX, id=doc["id"], document=doc)


def es_delete_document(product_id: str) -> None:
    """Delete a single product document. Safe/no-op if it doesn't exist
    (e.g. never synced, or a retry after a partial failure)."""
    es = get_es()
    es.options(ignore_status=404).delete(index=ES_INDEX, id=product_id)


def build_product_embedding(name: str, description: Optional[str]) -> List[float]:
    """Same document-side instruction prefix as scripts/seed_elasticsearch.py,
    so API-created products embed consistently with the seed dataset."""
    text = f"Represent this product for retrieval: {name}. {description or ''}"
    return get_embedding_model().encode(text, normalize_embeddings=True).tolist()


def build_review_embedding(title: Optional[str], text: Optional[str]) -> List[float]:
    """Same asymmetric document-side instruction-prefix convention as
    build_product_embedding, applied to review title+text."""
    combined = f"Represent this review for retrieval: {title or ''}. {text or ''}"
    return get_embedding_model().encode(combined, normalize_embeddings=True).tolist()


def query_products(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    logger.info("query_products called with filters: %s", filters)
    es = get_es()

    must_clauses = []
    filter_clauses = []

    if "search" in filters:
        must_clauses.append({"multi_match": {
            "query": filters["search"],
            "fields": ["name^2", "description"],
        }})
    if "category" in filters:
        filter_clauses.append({"term": {"category": filters["category"]}})
    if "price_min" in filters:
        filter_clauses.append({"range": {"price": {"gte": filters["price_min"]}}})
    if "price_max" in filters:
        filter_clauses.append({"range": {"price": {"lte": filters["price_max"]}}})
    if "rating_min" in filters:
        filter_clauses.append({"range": {"rating": {"gte": filters["rating_min"]}}})

    query: Dict[str, Any] = {"bool": {}}
    if must_clauses:
        query["bool"]["must"] = must_clauses
    if filter_clauses:
        query["bool"]["filter"] = filter_clauses
    if not must_clauses and not filter_clauses:
        query = {"match_all": {}}

    response = es.search(index=ES_INDEX, body={
        "size": 20,
        "query": query,
        "sort": [{"rating": "desc"}, {"reviews": "desc"}],
    })

    results = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        results.append({
            "id": src["id"],
            "name": src["name"],
            "price": src["price"],
            "originalprice": src.get("original_price"),
            "rating": src["rating"],
            "reviews": src["reviews"],
            "category_name": src["category"],
        })

    logger.info("query_products returned %d results", len(results))
    return results


def get_categories() -> List[Dict[str, Any]]:
    es = get_es()
    response = es.search(index=ES_INDEX, body={
        "size": 0,
        "aggs": {"categories": {"terms": {"field": "category", "size": 50}}},
    })
    buckets = response["aggregations"]["categories"]["buckets"]
    return [{"name": b["key"], "icon": "📦"} for b in buckets]


def semantic_search(query_text: str, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    from cache import get_cached_search, set_cached_search
    logger.info("semantic_search called: query='%s', limit=%d", query_text, limit)

    cached = get_cached_search(query_embedding)
    if cached is not None:
        search_cache_hits.inc()
        return cached[:limit]

    search_cache_misses.inc()
    es = get_es()
    candidates_size = max(20, limit * 4)

    t0 = time.perf_counter()
    response = es.search(index=ES_INDEX, body={
        "size": candidates_size,
        "query": {
            "bool": {
                "should": [
                    {"multi_match": {
                        "query": query_text,
                        "fields": ["name^2", "description"],
                        "boost": 0.5,
                    }},
                    {"knn": {
                        "field": "embedding",
                        "query_vector": query_embedding,
                        "num_candidates": 50,
                        "boost": 4.0,
                    }},
                ]
            }
        }
    })
    es_search_latency.observe(time.perf_counter() - t0)

    candidates = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        candidates.append({
            "id": src["id"],
            "name": src["name"],
            "description": src.get("description", ""),
            "price": src["price"],
            "originalprice": src.get("original_price"),
            "rating": src["rating"],
            "reviews": src["reviews"],
            "category_name": src["category"],
            "es_score": round(hit["_score"], 3),
        })

    if not candidates:
        logger.info("semantic_search: no candidates from ES")
        return []

    logger.info("Re-ranking %d candidates with cross-encoder...", len(candidates))
    reranker = get_reranker()
    pairs = [(query_text, f"{c['name']}. {c['description']}") for c in candidates]
    t1 = time.perf_counter()
    scores = reranker.predict(pairs).tolist()
    rerank_latency.observe(time.perf_counter() - t1)

    for candidate, score in zip(candidates, scores):
        candidate["similarity"] = round(score, 3)

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    results = candidates[:limit]

    for r in results:
        del r["description"]

    set_cached_search(query_text, query_embedding, results)
    logger.info("semantic_search returned %d results after reranking", len(results))
    return results


def semantic_search_reviews(query_text: str, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """Same hybrid BM25+kNN+rerank shape as semantic_search, targeting the
    reviews index and its own (separate) Redis cache namespace — see
    cache.py's get/set_cached_review_search for why this doesn't share
    the product-search cache."""
    from cache import get_cached_review_search, set_cached_review_search
    logger.info("semantic_search_reviews called: query='%s', limit=%d", query_text, limit)

    cached = get_cached_review_search(query_embedding)
    if cached is not None:
        search_cache_hits.inc()
        return cached[:limit]

    search_cache_misses.inc()
    es = get_es()
    candidates_size = max(20, limit * 4)

    t0 = time.perf_counter()
    response = es.search(index=REVIEWS_ES_INDEX, body={
        "size": candidates_size,
        "query": {
            "bool": {
                "should": [
                    {"multi_match": {
                        "query": query_text,
                        "fields": ["title^2", "text"],
                        "boost": 0.5,
                    }},
                    {"knn": {
                        "field": "embedding",
                        "query_vector": query_embedding,
                        "num_candidates": 50,
                        "boost": 4.0,
                    }},
                ]
            }
        }
    })
    es_search_latency.observe(time.perf_counter() - t0)

    candidates = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        candidates.append({
            "product_id": src["product_id"],
            "product_name": src.get("product_name", ""),
            "title": src.get("title", ""),
            "text": src.get("text", ""),
            "rating": src["rating"],
            "verified_purchase": src.get("verified_purchase", False),
            "helpful_vote": src.get("helpful_vote", 0),
            "es_score": round(hit["_score"], 3),
        })

    if not candidates:
        logger.info("semantic_search_reviews: no candidates from ES")
        return []

    logger.info("Re-ranking %d review candidates with cross-encoder...", len(candidates))
    reranker = get_reranker()
    pairs = [(query_text, f"{c['title']}. {c['text']}") for c in candidates]
    t1 = time.perf_counter()
    scores = reranker.predict(pairs).tolist()
    rerank_latency.observe(time.perf_counter() - t1)

    for candidate, score in zip(candidates, scores):
        candidate["similarity"] = round(score, 3)

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    results = candidates[:limit]

    set_cached_review_search(query_text, query_embedding, results)
    logger.info("semantic_search_reviews returned %d results after reranking", len(results))
    return results
