# Product Storage

The backend uses **Elasticsearch** as the product *search* store, combining full-text (BM25) search, exact-match filtering, and vector (kNN) search in a single index — this remains the read path for the `semantic_search`/`query_products`/`list_categories` agent tools. As of Phase 1 (see [Postgres (relational data)](#postgres-relational-data) below), **Postgres** also exists as a relational source of truth for products, users, addresses, and reviews, seeded independently from the same dataset. The two are not yet kept in sync with each other (no CDC — see the Notes section).

For the full field-level index mapping, Postgres table schemas, and Redis key schemas, see [docs/db-models.md](../docs/db-models.md). This document covers setup, initialization, and the tool contracts the agent uses.

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
- See [docs/cdc-reindex-pipeline.md](../docs/cdc-reindex-pipeline.md) for a planned (currently deferred — see issue #40) design to keep the index live-synced with an external source-of-truth database via Debezium/Kafka CDC.

## Postgres (relational data)

Phase 1 (issue #32) introduces Postgres as a relational store, separate from Elasticsearch. **ES remains the read path for the agent's search tools in Phase 1** — Postgres is additive, not yet a replacement for `query_products`/`semantic_search`. Whether product reads ever move onto Postgres is a future decision, not assumed here.

**Tables**: `products`, `product_images` (1:N), `users`, `addresses` (1:N), `reviews` (1:N via `product_id`). See [docs/db-models.md](../docs/db-models.md) for the full field-level schema. `product_variants` was considered and explicitly dropped — the seed dataset has no real variant grouping (see issue #32). Cart/orders/payments are Phase 2, not yet built.

**Async by design**: unlike `search.py`/`cache.py` (currently sync), the Postgres layer (`db.py`, `models_db.py`) uses async SQLAlchemy (`asyncpg`, `AsyncEngine`, `AsyncSession`) from the start. This is deliberate: issue #41 found the backend can't serve concurrent requests today (fully blocking event loop, single uvicorn worker) and scoped a full async rewrite as the fix. Rather than write this layer sync and redo it later, it's async from day one — the rest of the codebase remains sync until #41 is picked up separately.

**Connection**: `DATABASE_URL` (see `app/config.py`), using the `postgresql+asyncpg://` scheme (required for `create_async_engine` to select the asyncpg driver). Local default: `postgresql+asyncpg://postgres:postgres@localhost:5432/ecommerce`. In `docker-compose.yml`, the `postgres` service maps host port **5433** (not 5432) to avoid conflicting with a native Postgres a developer might already have running locally — the container-internal port is still 5432, so backend↔postgres networking inside Docker is unaffected, only host-facing tools (e.g. `psql` from your own machine) need port 5433.

For a hosted Supabase/Neon instance (anything beyond local dev — see issue #32 for why), override `DATABASE_URL` with the `postgresql+asyncpg://` scheme in that environment's env config. Note: some hosted providers front connections through a transaction-mode pooler (e.g. Supabase's pgbouncer) that doesn't support all wire-protocol features `asyncpg` relies on for prepared statements — use the provider's direct (non-pooled) connection string if you hit errors.

**Migrations**: Alembic, run automatically at backend startup (`run_migrations()` in `db.py`, called from `app/main.py` alongside `init_es_index()`/`init_cache_index()`) — applies any pending migration on every boot, safe/idempotent since it's a no-op once the schema is at head. This differs from `init_es_index()`'s `create_all()`-style "create if missing" pattern: `alembic upgrade head` actually applies schema *changes*, not just initial creation, so it stays correct as the schema evolves. To run migrations manually (e.g. before running the seed script, without booting the full app):

```bash
docker compose up -d postgres
cd backend
alembic upgrade head   # optional — the app also runs this automatically on boot
```

**Seeding**: `python scripts/seed_postgres.py` (same invocation shape as `index_products.py` — run from inside the backend container, or via `docker compose exec backend python scripts/seed_postgres.py`). Sources:
- `products`/`product_images`: same `McAuley-Lab/Amazon-Reviews-2023` dataset and category/product-count limits as `index_products.py`, re-read independently (not dependent on ES being populated first). Product filtering is kept in sync with `index_products.py` via a shared `_parse_amazon_product()`-style helper so `Product.id` values overlap 1:1 with the ES `_id` space.
- `reviews`: real review text from the same dataset's `raw_review_categories/{Category}.jsonl` files, joined to seeded products via `parent_asin`.
- `users`/`addresses`: **synthetic**, generated with Faker — no real user dataset exists in the source data (review `user_id` values are opaque anonymized hashes, not usable as real user records).

Like `index_products.py`, this wipes and recreates the schema for a clean run each time — not incremental.

**Sync note**: `products`/`product_images`/`reviews` in Postgres and the ES `products` index are currently two independently-seeded, non-synced copies of overlapping data (both scripts source from the same dataset but run independently, and can be run in either order). No CDC/sync exists between them (see `docs/cdc-reindex-pipeline.md`, tracked as issue #40 and explicitly deferred — no multi-writer/multi-consumer need justifies it today).

As of #49, this applies only to the two seed scripts' initial data. Writes made **through the product API routes** (`app/routes/products.py`) stay in sync: each route writes Postgres first, then performs the equivalent single-document ES write (`es_upsert_document`/`es_delete_document`), regenerating the BGE embedding on every create/update. This is lightweight and non-transactional — if the ES write fails after a successful Postgres commit, the request still returns success and the product is left stale in ES with no retry queue. See #40 (deferred) for the CDC-based alternative that would close this gap.

Phase 2 (cart, orders, payments — see issue tracker) will add tables to this same Postgres database using the same `Base`/`get_session()` pattern in `db.py`.

### Product write API (#49)

`POST /products`, `PATCH /products/{id}`, and `DELETE /products/{id}` (`app/routes/products.py`) are the only write path into `products` outside the seed scripts. Create generates a new `id` via `uuid.uuid4().hex` (distinct in shape from the seed dataset's `parent_asin` ids — see `docs/db-models.md`). Update is a partial update (`exclude_unset=True` semantics — an explicit `null` clears a field, an omitted field is left untouched) and unconditionally regenerates the product's embedding on every call, since `content_hash`-based change detection isn't wired up yet. Delete cascades to `product_images`/`reviews` in Postgres (FK `ondelete="CASCADE"`) but is blocked (`IntegrityError` → `500`) if the product has order history, via `order_items.product_id`'s `ondelete="RESTRICT"`.
