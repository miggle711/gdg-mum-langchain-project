import json
import logging
import math
import redis
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from elasticsearch import Elasticsearch
from app.config import settings

logger = logging.getLogger(__name__)

ELASTICSEARCH_URL = settings.elasticsearch_url
ES_INDEX = "products"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 768

# ES client singleton
_es: Optional[Elasticsearch] = None

# Cross-encoder singleton — loaded once at startup
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder reranker: %s", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker loaded.")
    return _reranker


def get_es() -> Elasticsearch:
    global _es
    if _es is None:
        _es = Elasticsearch(ELASTICSEARCH_URL)
    return _es


def init_es_index():
    """Create the ES products index with hybrid search mapping if it doesn't exist."""
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


def es_bulk_index(documents: List[Dict[str, Any]]) -> None:
    """Bulk index a list of product documents into Elasticsearch."""
    from elasticsearch.helpers import bulk
    es = get_es()
    logger.info("Bulk indexing %d documents into ES index '%s'...", len(documents), ES_INDEX)

    def _actions():
        for doc in documents:
            yield {"_index": ES_INDEX, "_id": doc["id"], "_source": doc}

    bulk(es, _actions())
    logger.info("Bulk indexing complete.")


def query_products(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter products in Elasticsearch by exact criteria: category, price, rating, keyword."""
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

    query = {"bool": {}}
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
    """Get distinct categories from Elasticsearch via aggregation."""
    es = get_es()
    response = es.search(index=ES_INDEX, body={
        "size": 0,
        "aggs": {"categories": {"terms": {"field": "category", "size": 50}}},
    })
    buckets = response["aggregations"]["categories"]["buckets"]
    return [{"name": b["key"], "icon": "📦"} for b in buckets]


def semantic_search(query_text: str, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """Hybrid search with cross-encoder re-ranking and Redis semantic cache.

    1. Check Redis for a cached result with a semantically similar embedding
    2. On miss: ES retrieves top 20 candidates via BM25 + dense vector (recall)
    3. Cross-encoder re-scores each candidate against the query (precision)
    4. Cache and return top `limit` after re-ranking
    """
    logger.info("semantic_search called: query='%s', limit=%d", query_text, limit)

    cached = get_cached_search(query_embedding)
    if cached is not None:
        return cached[:limit]

    es = get_es()
    candidates_size = max(20, limit * 4)

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
    scores = reranker.predict(pairs).tolist()

    for candidate, score in zip(candidates, scores):
        candidate["similarity"] = round(score, 3)

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    results = candidates[:limit]

    for r in results:
        del r["description"]

    set_cached_search(query_text, query_embedding, results)
    logger.info("semantic_search returned %d results after reranking", len(results))
    return results


# Redis connection pool — created once, reused across all requests
_redis_pool = redis.ConnectionPool.from_url(
    settings.redis_url,
    decode_responses=True,
    max_connections=20,
)

CONVERSATION_TTL_SECONDS = settings.conversation_ttl_seconds
SEARCH_CACHE_TTL_SECONDS = settings.search_cache_ttl_seconds
SEARCH_CACHE_SIMILARITY_THRESHOLD = settings.search_cache_similarity_threshold


def _get_redis() -> redis.Redis:
    return redis.Redis(connection_pool=_redis_pool)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_cached_search(query_embedding: List[float]) -> Optional[List[Dict[str, Any]]]:
    """Check Redis for a semantically similar cached search result."""
    r = _get_redis()
    for key in r.scan_iter("search_cache:*"):
        entry = r.get(key)
        if entry is None:
            continue
        data = json.loads(entry)
        similarity = _cosine_similarity(query_embedding, data["embedding"])
        if similarity >= SEARCH_CACHE_SIMILARITY_THRESHOLD:
            logger.info("Search cache HIT (similarity=%.3f, key=%s)", similarity, key)
            return data["results"]
    return None


def set_cached_search(query_text: str, query_embedding: List[float], results: List[Dict[str, Any]]) -> None:
    """Store search results in Redis keyed by a hash of the query text."""
    r = _get_redis()
    key = f"search_cache:{abs(hash(query_text))}"
    r.set(key, json.dumps({"embedding": query_embedding, "results": results}), ex=SEARCH_CACHE_TTL_SECONDS)
    logger.info("Search result cached under key=%s (TTL=%ds)", key, SEARCH_CACHE_TTL_SECONDS)


def save_messages(conversation_id: str, messages: List[BaseMessage]) -> None:
    """Serialize and save a conversation's messages to Redis."""
    r = _get_redis()
    key = f"conversation:{conversation_id}"
    r.set(key, json.dumps(messages_to_dict(messages)), ex=CONVERSATION_TTL_SECONDS)
    logger.info("Saved %d messages for conversation '%s'", len(messages), conversation_id)


def load_messages(conversation_id: str) -> List[BaseMessage]:
    """Load and deserialize a conversation's messages from Redis. Returns [] if not found."""
    r = _get_redis()
    key = f"conversation:{conversation_id}"
    data = r.get(key)
    if data is None:
        logger.info("No session found in Redis for conversation '%s', starting fresh.", conversation_id)
        return []
    messages = messages_from_dict(json.loads(data))
    logger.info("Loaded %d messages for conversation '%s' from Redis.", len(messages), conversation_id)
    return messages
