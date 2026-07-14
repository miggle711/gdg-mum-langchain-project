# Data Models

Product *search* data lives in an Elasticsearch document index; conversation state and the semantic search cache live in Redis as key-value/vector entries. As of Phase 1 (issue #32), a **Postgres** relational database also exists for products, users, addresses, and reviews — see [Postgres: relational tables](#postgres-relational-tables) below. This document is the field-level reference for all three; see [backend/DATABASE.md](../backend/DATABASE.md) for setup/usage context.

## Elasticsearch: `products` index

Defined in `backend/search.py`'s `init_es_index()`.

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `keyword` | Also used as the document's `_id` (see `es_bulk_index()`) |
| `name` | `text` (english analyzer) | Full-text/BM25 search field, boosted 2x in `multi_match` queries |
| `description` | `text` (english analyzer) | Full-text search field; truncated to 500 chars at index time |
| `category` | `keyword` | Exact-match only — used for `term` filters in `query_products` and the `list_categories` aggregation |
| `price` | `float` | Range filters, sort key |
| `original_price` | `float` | Optional; nullable |
| `rating` | `float` | Range filters (`rating_min`), primary sort key for `query_products` |
| `reviews` | `integer` | Secondary sort key for `query_products` |
| `image` | `keyword`, `index: false` | Stored for display only — never searched or filtered |
| `embedding` | `dense_vector`, dims 768, `index: true`, `similarity: cosine` | BGE (`BAAI/bge-base-en-v1.5`) embedding of `name + description`; powers kNN in `semantic_search` |

No relations — this is a flat document model. There is no separate "categories" entity; `category` is just a keyword field on each product, and the distinct set of values is computed on demand via a terms aggregation (`get_categories()` in `search.py`), not stored separately.

## Redis: keys and schemas

Redis Stack is used for three distinct purposes, using different key prefixes and a mix of string/hash/vector-index storage. Connection and TTLs are configured in `backend/app/config.py`.

### Conversation history — `conversation:{conversation_id}`

- **Type**: string (JSON)
- **Written/read by**: `backend/conversations.py` (`save_messages`, `load_messages`)
- **Value**: `langchain_core.messages` serialized via `messages_to_dict()` — a JSON list of `{type, data: {content, ...}}` objects
- **TTL**: `conversation_ttl_seconds` (default 24h) — conversations expire, they are not kept forever

### Conversation summary — `conversation:{conversation_id}:summary`

- **Type**: string
- **Written/read by**: `conversations.py` (`save_summary`, `load_summary`)
- **Value**: plain-text LLM-generated summary of older turns
- **Written when**: message count exceeds `conversation_summary_threshold` (default 20 messages / 10 turns) — see `maybe_summarise()`. Only the older half of history is summarized; the most recent half is kept verbatim and re-saved under the main `conversation:{id}` key.
- **TTL**: same `conversation_ttl_seconds` as the conversation itself

### Semantic search cache — `search_cache:{hash}` + vector index `idx:search_cache`

- **Type**: hash, indexed by a RediSearch HNSW vector index
- **Written/read by**: `backend/cache.py` (`get_cached_search`, `set_cached_search`, `init_cache_index`)
- **Key**: `search_cache:{abs(hash(query_text))}` — note this uses Python's built-in (process-salted) `hash()`, so it's not stable across process restarts; lookups always go through the vector index, not this key directly

**Fields:**

| Field | Type | Notes |
| --- | --- | --- |
| `embedding` | raw `FLOAT32` bytes | The query embedding, packed via `struct.pack`; the field indexed by HNSW (dim 768, cosine) |
| `results` | text (JSON) | The cached, reranked `semantic_search()` result list |

- **Lookup**: KNN query for the single nearest cached embedding; a result only counts as a cache hit if cosine distance is within `search_cache_similarity_threshold` (default 0.92 similarity, i.e. distance ≤ 0.08)
- **TTL**: `search_cache_ttl_seconds` (default 6h) per entry
- **Invalidation**: none beyond TTL expiry — updating a product does not proactively invalidate matching cache entries (see the "Notes" section in [backend/DATABASE.md](../backend/DATABASE.md))

## Postgres: relational tables

Defined in `backend/models_db.py` (SQLAlchemy ORM), migrated via Alembic (`backend/alembic/`). Async engine/session (`db.py`) — see [backend/DATABASE.md](../backend/DATABASE.md#postgres-relational-data) for why.

### `products`

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `String` (PK) | `parent_asin` from the source dataset — matches the ES document `_id`. Products created via `POST /products` (#49) instead get a `uuid.uuid4().hex` id — the two id shapes (short alphanumeric ASINs vs. 32-char hex UUIDs) coexist in this column with no collision risk. |
| `name` | `Text` | |
| `description` | `Text`, nullable | |
| `category` | `String`, indexed | Plain column, not a separate entity — same reasoning as the ES `category` field (see below) |
| `price` | `Float` | |
| `original_price` | `Float`, nullable | |
| `rating` | `Float`, nullable | |
| `reviews` | `Integer`, nullable, default 0 | Denormalized review **count**, carried over from the ES schema — distinct from the `reviews` table below. Kept as a plain column (not computed live) since nothing reads it from Postgres at request time yet in Phase 1. |
| `image` | `Text`, nullable | |
| `content_hash` | `String`, nullable | Not populated in Phase 1 seed data — reserved for a future CDC pipeline's re-embed/skip logic (issue #40, currently deferred) |
| `created_at` | `DateTime` | |

Relationships: `images` (1:N → `product_images`, cascade delete), `reviews_rel` (1:N → `reviews`, cascade delete — named `reviews_rel` rather than `reviews` specifically to avoid colliding with the `reviews` count column above).

### `product_images`

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `Integer` (PK, autoincrement) | |
| `product_id` | `String`, FK → `products.id` (`ON DELETE CASCADE`), indexed | |
| `image_url` | `Text` | |
| `position` | `Integer`, default 0 | Preserves the source dataset's image ordering |

### `users`

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `Integer` (PK, autoincrement) | |
| `email` | `String`, unique, indexed | |
| `name` | `String` | |
| `created_at` | `DateTime` | |

**Synthetic data** (Faker-generated) — no real user dataset exists; the source dataset's review `user_id` values are opaque anonymized hashes, not usable as real user records. Relationship: `addresses` (1:N, cascade delete).

### `addresses`

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `Integer` (PK, autoincrement) | |
| `user_id` | `Integer`, FK → `users.id` (`ON DELETE CASCADE`), indexed | |
| `street` | `String` | |
| `city` | `String` | |
| `state` | `String`, nullable | |
| `zip_code` | `String` | |
| `country` | `String` | |

Synthetic, same as `users` — a user can have 1-3 addresses in the seed data.

### `reviews`

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `Integer` (PK, autoincrement) | |
| `product_id` | `String`, FK → `products.id` (`ON DELETE CASCADE`), indexed | |
| `rating` | `Float` | |
| `title` | `Text`, nullable | |
| `text` | `Text`, nullable | The actual review text — candidate for future BGE embedding to power semantic review search, not embedded in Phase 1 |
| `verified_purchase` | `Boolean`, default false | |
| `helpful_vote` | `Integer`, default 0 | |
| `timestamp` | `BigInteger`, nullable | Raw epoch-ms from the source dataset, kept as-is |

**Real data** (unlike `users`/`addresses`) — sourced from `McAuley-Lab/Amazon-Reviews-2023`'s `raw_review_categories/{Category}.jsonl` files, joined to seeded products via `parent_asin` = `products.id`.

### Dropped from Phase 1 scope: `product_variants`

Considered (product_id FK, sku, attributes, price, stock) but dropped — confirmed the source dataset has no real variant grouping (no products share a `parent_asin`/variant relationship). Would have meant inert schema seeded with fabricated pass-through rows and zero real consumers. See issue #32 for full reasoning; revisit only if real variant data or a concrete need shows up, likely alongside Phase 2 (cart/orders).

## Config reference

All TTLs, thresholds, and connection URLs above are defined once in `backend/app/config.py` (`Settings`, loaded via `pydantic-settings` from environment variables / `.env`) — treat that file as the source of truth if any of the defaults listed here change.
