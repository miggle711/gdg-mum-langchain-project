import json
import logging
import struct
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from typing import List, Dict, Any, Optional
from app.config import settings
from search import EMBEDDING_DIM

logger = logging.getLogger(__name__)

SEARCH_CACHE_TTL_SECONDS = settings.search_cache_ttl_seconds
SEARCH_CACHE_SIMILARITY_THRESHOLD = settings.search_cache_similarity_threshold

CACHE_INDEX = "idx:search_cache"
CACHE_PREFIX = "search_cache:"

# String pool — for conversation history
_redis_pool = redis.ConnectionPool.from_url(
    settings.redis_url,
    decode_responses=True,
    max_connections=20,
)

# Binary pool — for vector cache (raw bytes can't be decoded as strings)
_redis_binary_pool = redis.ConnectionPool.from_url(
    settings.redis_url,
    decode_responses=False,
    max_connections=10,
)


def _get_redis() -> redis.Redis:
    return redis.Redis(connection_pool=_redis_pool)


def _get_redis_binary() -> redis.Redis:
    return redis.Redis(connection_pool=_redis_binary_pool)


def _embedding_to_bytes(embedding: List[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def init_cache_index() -> None:
    r = _get_redis_binary()
    try:
        r.ft(CACHE_INDEX).info()
        logger.info("Redis cache vector index '%s' already exists.", CACHE_INDEX)
        return
    except Exception:
        pass

    logger.info("Creating Redis cache vector index '%s'...", CACHE_INDEX)
    r.ft(CACHE_INDEX).create_index(
        fields=[
            VectorField(
                "embedding",
                "HNSW",
                {"TYPE": "FLOAT32", "DIM": EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE", "M": 16, "EF_CONSTRUCTION": 200},
            ),
            TextField("results"),
        ],
        definition=IndexDefinition(prefix=[CACHE_PREFIX], index_type=IndexType.HASH),
    )
    logger.info("Redis cache vector index created.")


def get_cached_search(query_embedding: List[float]) -> Optional[List[Dict[str, Any]]]:
    r = _get_redis_binary()
    query_bytes = _embedding_to_bytes(query_embedding)
    threshold = 1.0 - SEARCH_CACHE_SIMILARITY_THRESHOLD

    q = (
        Query("*=>[KNN 1 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("score", "results")
        .dialect(2)
    )
    results = r.ft(CACHE_INDEX).search(q, query_params={"vec": query_bytes})

    if not results.docs:
        return None

    doc = results.docs[0]
    distance = float(doc.score)
    if distance > threshold:
        return None

    logger.info("Search cache HIT (cosine distance=%.3f, key=%s)", distance, doc.id)
    return json.loads(doc.results)


def set_cached_search(query_text: str, query_embedding: List[float], results: List[Dict[str, Any]]) -> None:
    r = _get_redis_binary()
    key = f"{CACHE_PREFIX}{abs(hash(query_text))}"
    pipe = r.pipeline()
    pipe.hset(key, mapping={
        "embedding": _embedding_to_bytes(query_embedding),
        "results": json.dumps(results),
    })
    pipe.expire(key, SEARCH_CACHE_TTL_SECONDS)
    pipe.execute()
    logger.info("Search result cached under key=%s (TTL=%ds)", key, SEARCH_CACHE_TTL_SECONDS)
