# Product Storage

The backend uses **Elasticsearch** as the product catalog store, combining full-text (BM25) search, exact-match filtering, and vector (kNN) search in a single index. There is no relational database for products — the catalog is entirely document-based.

For the full field-level index mapping and Redis key schemas, see [docs/db-models.md](../docs/db-models.md). This document covers setup, initialization, and the tool contracts the agent uses.

## Index

- **Name**: `products` (see `ES_INDEX` in `search.py`)
- **Created by**: `init_es_index()` in `search.py`, called once at backend startup (`app/main.py`). It's idempotent — if the index already exists, it's left untouched.
- **Connection**: configured via `ELASTICSEARCH_URL` (see `app/config.py`), defaults to `http://localhost:9200`.

Each product document has: `id`, `name`, `description` (both full-text, English analyzer), `category` (keyword/exact-match), `price`, `original_price`, `rating`, `reviews` (numeric), `image` (stored but not indexed), and `embedding` (768-dim dense vector, cosine similarity — a `BAAI/bge-base-en-v1.5` embedding of the product's name + description).

## Loading Sample Data

The index is created empty. To populate it with a sample catalog, run the indexing script (from inside the backend container):

```bash
docker compose exec backend python scripts/index_products.py
```

This script (`backend/scripts/index_products.py`):

1. Deletes and recreates the `products` index for a clean run.
2. Streams product metadata from the `McAuley-Lab/Amazon-Reviews-2023` dataset for 4 categories (`Sports_and_Outdoors`, `Electronics`, `Home_and_Kitchen`, `Toys_and_Games`), taking up to 500 products per category (~2k products total).
3. Generates a BGE embedding for each product using the instruction prefix `"Represent this product for retrieval: {name}. {description}"`, batched (256 at a time) for efficiency.
4. Bulk-indexes everything into Elasticsearch via `es_bulk_index()`.

This is a one-time/manual step, not run automatically on backend startup — only the empty index schema is created automatically.

## Product Query Tools

The LangChain agent (`app/agent.py`) has access to three tools (defined in `tools.py`, backed by `search.py`):

### `semantic_search(query, limit=5)`

Natural-language product search for vague or descriptive queries.

```text
Input: {"query": "something cozy for winter", "limit": 5}
Returns: The most semantically similar products, each with a similarity score
```

Pipeline: checks a Redis semantic cache first (embedding KNN, cosine similarity) → on a miss, runs a hybrid Elasticsearch query (BM25 `multi_match` boosted 0.5 + kNN on `embedding` boosted 4.0) → reranks the candidates with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) → caches and returns the top results. See the "Search pipeline" diagram in the [root README](../README.md) for the full flow.

### `query_products(search=None, category=None, price_min=None, price_max=None, rating_min=None)`

Exact-filter product search — no semantic ranking, no reranking.

```text
Input: {"category": "Electronics", "price_max": 100, "rating_min": 4.0}
Returns: Products in Electronics under $100 with a rating of 4.0+, sorted by rating desc then reviews desc
```

`category` must match an indexed category name exactly (it's a `keyword` field) — this is why the agent is instructed to call `list_categories` first when filtering by category.

### `list_categories()`

Returns all distinct category values in the index (via an Elasticsearch terms aggregation), so the agent can use exact names instead of guessing.

```text
Returns: [{"name": "Electronics", "icon": "📦"}, ...]
```

## Testing Queries Directly

To exercise the search layer without going through the chat/agent flow:

```python
from search import query_products, semantic_search, get_categories

# Exact filters
results = query_products({"category": "Electronics", "price_max": 100})

# Semantic search (requires a precomputed query embedding — see tools.py for how
# the agent generates one with the BGE model and the retrieval instruction prefix)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embedding = model.encode(
    "Represent this sentence for searching relevant passages: cozy winter blanket",
    normalize_embeddings=True,
).tolist()
results = semantic_search("cozy winter blanket", embedding, limit=5)

# Categories
categories = get_categories()
```

## Notes

- The Redis semantic search cache (`cache.py`) sits in front of `semantic_search()` only — `query_products()` and `list_categories()` always hit Elasticsearch directly.
- Cache entries expire after `SEARCH_CACHE_TTL_SECONDS` (default 6h, see `app/config.py`); there's no automatic invalidation when the underlying catalog changes, so a stale product update may not be reflected in cached semantic search results until the TTL elapses.
- See [docs/cdc-reindex-pipeline.md](../docs/cdc-reindex-pipeline.md) for a planned (not yet implemented) design to keep the index live-synced with an external source-of-truth database via Debezium/Kafka CDC.
