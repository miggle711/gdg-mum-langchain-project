# Product Storage

The backend uses **Elasticsearch** as the product/review *search* store, combining full-text (BM25) search, exact-match filtering, and vector (kNN) search — this remains the read path for the `semantic_search`/`query_products`/`list_categories`/`search_reviews` agent tools. **Postgres** is the relational source of truth for products, reviews, users, addresses, carts, and orders (see [Postgres (relational data)](#postgres-relational-data) below). As of #51, Elasticsearch is seeded *from* Postgres, not independently from the raw dataset — see [Loading Sample Data](#loading-sample-data).

For the full field-level index mapping, Postgres table schemas, and Redis key schemas, see [docs/db-models.md](../docs/db-models.md). This document covers setup, initialization, and the tool contracts the agent uses.

## Indices

- **`products`** (see `ES_INDEX` in `search.py`): `id`, `name`, `description` (both full-text, English analyzer), `category` (keyword/exact-match), `price`, `original_price`, `rating`, `reviews` (numeric), `image` (stored but not indexed), and `embedding` (768-dim dense vector, cosine similarity — a `BAAI/bge-base-en-v1.5` embedding of the product's name + description).
- **`reviews`** (see `REVIEWS_ES_INDEX` in `search.py`): `id`, `product_id` (keyword), `title`/`text` (both full-text, English analyzer), `rating`, `verified_purchase`, `helpful_vote`, and `embedding` (same 768-dim/cosine shape, a BGE embedding of the review's title + text).

Both are **created by** `init_es_index()`/`init_reviews_index()` in `search.py`, called once at backend startup (`app/main.py`). Both are idempotent — if an index already exists, it's left untouched. **Connection**: configured via `ELASTICSEARCH_URL` (see `app/config.py`), defaults to `http://localhost:9200`.

## Loading Sample Data

Startup only creates the empty index schemas — populating them with data is a manual, two-step process, since Elasticsearch is seeded from Postgres, not the raw dataset directly:

```bash
docker compose exec backend python scripts/seed_postgres.py
docker compose exec backend python scripts/seed_elasticsearch.py
```

**Step 1** (`scripts/seed_postgres.py`, unchanged by #51) streams product/review metadata from the `McAuley-Lab/Amazon-Reviews-2023` dataset (4 categories, up to 500 products/category, ~200 reviews total — see #69 for the review-volume cap) and inserts it into Postgres.

**Step 2** (`scripts/seed_elasticsearch.py`, replaces the old `index_products.py`):

1. Deletes and recreates both the `products` and `reviews` ES indices for a clean run.
2. Reads `Product` and `Review` rows back out of Postgres (no HF dataset dependency of its own — requires step 1 to have run first).
3. Generates a BGE embedding for each product (`"Represent this product for retrieval: {name}. {description}"`) and each review (`"Represent this review for retrieval: {title}. {text}"`), batched (256 at a time).
4. Bulk-indexes everything into their respective ES indices.

This is a one-time/manual step, not run automatically on backend startup. This is an intentional change from earlier versions of this project (issue #39), where product seeding read directly from the HF dataset and could safely auto-run as a fast startup check — now that ES depends on Postgres data existing, auto-seeding at boot would mean blocking every startup on a slow HF download + Postgres write + embedding pass, and would add a second external-network failure mode alongside #52's existing ES-unreachable-at-boot issue.

## Product and Review Query Tools

The LangChain agent (`app/agent.py`) has access to four tools (defined in `tools.py`, backed by `search.py`):

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

### `search_reviews(query, limit=5)`

Natural-language search over review text — for questions about what customers say/think, not for finding products themselves (#51). Same hybrid BM25+kNN+rerank pipeline as `semantic_search`, but targets the `reviews` index (`title`/`text` fields instead of `name`/`description`) and its own separate Redis cache namespace (`idx:review_search_cache` — deliberately not shared with product search's cache, since a shared vector-only cache has no query-type discriminator and could return the wrong kind of cached result for a similar-looking query).

```text
Input: {"query": "battery life complaints", "limit": 5}
Returns: Matching reviews (product_id, rating, title, text, similarity score)
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

**Async throughout (#41)**: the Postgres layer (`db.py`, `models_db.py`) was async from the start (`asyncpg`, `AsyncEngine`, `AsyncSession`), and `search.py`/`cache.py`/`conversations.py` were converted to match (`AsyncElasticsearch`, `redis.asyncio`) once issue #41 found the backend couldn't serve concurrent requests — route handlers were `async def` but the actual ES/Redis clients underneath were sync, blocking the single event loop for the full duration of every Gemini/ES/Redis round-trip. ES/Redis index initialization at boot now runs inside `app/main.py`'s FastAPI `lifespan` context manager rather than at plain module-import time — this matters because the async clients bind to whichever event loop they first run on, and a module-level `asyncio.run(...)` call would create connections bound to a throwaway loop that's already closed by the time uvicorn's real loop starts serving requests. `run_migrations()` is the one exception that stays at plain import time, since Alembic drives its own event loop internally and can't run inside one that's already active.

**Connection**: `DATABASE_URL` (see `app/config.py`), using the `postgresql+asyncpg://` scheme (required for `create_async_engine` to select the asyncpg driver). Local default: `postgresql+asyncpg://postgres:postgres@localhost:5432/ecommerce`. In `docker-compose.yml`, the `postgres` service maps host port **5433** (not 5432) to avoid conflicting with a native Postgres a developer might already have running locally — the container-internal port is still 5432, so backend↔postgres networking inside Docker is unaffected, only host-facing tools (e.g. `psql` from your own machine) need port 5433.

For a hosted Supabase/Neon instance (anything beyond local dev — see issue #32 for why), override `DATABASE_URL` with the `postgresql+asyncpg://` scheme in that environment's env config. Note: some hosted providers front connections through a transaction-mode pooler (e.g. Supabase's pgbouncer) that doesn't support all wire-protocol features `asyncpg` relies on for prepared statements — use the provider's direct (non-pooled) connection string if you hit errors.

**Migrations**: Alembic, run automatically at backend startup (`run_migrations()` in `db.py`, called from `app/main.py` alongside `init_es_index()`/`init_cache_index()`) — applies any pending migration on every boot, safe/idempotent since it's a no-op once the schema is at head. This differs from `init_es_index()`'s `create_all()`-style "create if missing" pattern: `alembic upgrade head` actually applies schema *changes*, not just initial creation, so it stays correct as the schema evolves. To run migrations manually (e.g. before running the seed script, without booting the full app):

```bash
docker compose up -d postgres
cd backend
alembic upgrade head   # optional — the app also runs this automatically on boot
```

**Seeding**: `python scripts/seed_postgres.py` (run from inside the backend container, or via `docker compose exec backend python scripts/seed_postgres.py`). Sources:
- `products`/`product_images`: `McAuley-Lab/Amazon-Reviews-2023` dataset, 4 categories, up to 500 products/category.
- `reviews`: real review text from the same dataset's `raw_review_{category}` files, joined to seeded products via `parent_asin`, capped at ~200 total (see #69 — this cap is intentionally low today and can be raised).
- `users`/`addresses`: **synthetic**, generated with Faker — no real user dataset exists in the source data (review `user_id` values are opaque anonymized hashes, not usable as real user records).

This wipes and recreates the schema for a clean run each time — not incremental.

**Sync note (updated by #51)**: Postgres is now the single source of truth Elasticsearch is seeded from — run `scripts/seed_postgres.py` first, then `scripts/seed_elasticsearch.py` (see [Loading Sample Data](#loading-sample-data)). This replaces the earlier setup where `index_products.py` and `seed_postgres.py` independently streamed from the same HF dataset with no relationship to each other. There is still no CDC/live-sync between the two stores after seeding — a later Postgres write only reaches ES if it goes through a code path that explicitly does so (see [Product write API (#49)](#product-write-api-49) below for the one that does). See `docs/cdc-reindex-pipeline.md` (issue #40, deferred) for the live-sync alternative.

As of #49, this applies only to the two seed scripts' initial data. Writes made **through the product API routes** (`app/routes/products.py`) stay in sync: each route writes Postgres first, then performs the equivalent single-document ES write (`es_upsert_document`/`es_delete_document`), regenerating the BGE embedding on every create/update. This is lightweight and non-transactional — if the ES write fails after a successful Postgres commit, the request still returns success and the product is left stale in ES with no retry queue. See #40 (deferred) for the CDC-based alternative that would close this gap.

Phase 2 (cart, orders, payments — see issue tracker) will add tables to this same Postgres database using the same `Base`/`get_session()` pattern in `db.py`.

### Product write API (#49)

`POST /products`, `PATCH /products/{id}`, and `DELETE /products/{id}` (`app/routes/products.py`) are the only write path into `products` outside the seed scripts. Create generates a new `id` via `uuid.uuid4().hex` (distinct in shape from the seed dataset's `parent_asin` ids — see `docs/db-models.md`). Update is a partial update (`exclude_unset=True` semantics — an explicit `null` clears a field, an omitted field is left untouched) and unconditionally regenerates the product's embedding on every call, since `content_hash`-based change detection isn't wired up yet. Delete cascades to `product_images`/`reviews` in Postgres (FK `ondelete="CASCADE"`) but is blocked (`IntegrityError` → `500`) if the product has order history, via `order_items.product_id`'s `ondelete="RESTRICT"`.
