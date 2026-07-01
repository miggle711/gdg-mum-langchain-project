from prometheus_client import Counter, Histogram

search_cache_hits = Counter(
    "search_cache_hits_total",
    "Number of Redis semantic cache hits",
)

search_cache_misses = Counter(
    "search_cache_misses_total",
    "Number of Redis semantic cache misses",
)

rerank_latency = Histogram(
    "rerank_duration_seconds",
    "Cross-encoder reranking latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

es_search_latency = Histogram(
    "es_search_duration_seconds",
    "Elasticsearch hybrid search latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
