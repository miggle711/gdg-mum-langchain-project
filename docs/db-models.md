# Data Models

Product *search* data lives in an Elasticsearch document index; conversation state and the semantic search cache live in Redis as key-value/vector entries. A **Postgres** relational database (issue #32) holds products, users, addresses, reviews, carts/orders/payments (#49+), user preferences (#54), and the auth `password_hash` column (#82) — see [Postgres: relational tables](#postgres-relational-tables) below. This document is the field-level reference for all three; see [backend/DATABASE.md](../backend/DATABASE.md) for setup/usage context.

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
- **`conversation_id` shape** (#82, #54): for an authenticated request (valid JWT), `user:{user_id}` — so a logged-in user's history is keyed by their durable, unspoofable identity rather than the client-supplied `session_id`, and follows them across separate chat sessions. For a guest, the raw client-supplied `session_id`, **except** a guest-supplied `session_id` that itself starts with the reserved `user:` prefix is re-namespaced under `guest:` (`app/routes/chat.py`'s `_conversation_key`) — closes a spoofing path where a guest could otherwise send `session_id="user:55"` to read/write a real user's conversation history with no valid JWT.

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
| `email` | `String`, unique, indexed | For shadow/guest users (see below), a synthetic `session-{session_id}@shadow.local` address, not a real email |
| `name` | `String` | |
| `password_hash` | `String`, nullable | `NULL` = guest/shadow user; a bcrypt hash = a real signed-up account (#82). This single column is the entire discriminator — both kinds of identity are the same table/row shape, so no downstream code (cart, orders, preferences) needs to know which kind it's dealing with. |
| `created_at` | `DateTime` | |

Two distinct ways rows get created here, both converging on this one table:

- **Seed data**: synthetic (Faker-generated) — no real user dataset exists; the source dataset's review `user_id` values are opaque anonymized hashes, not usable as real user records. These rows never get a `password_hash`.
- **Real accounts** (#82): `POST /auth/signup` creates a row with a real `email`/`name` and a bcrypt `password_hash`. **Guest/shadow accounts** (pre-existing, unauthenticated flow): `session_identity.py`'s `get_or_create_shadow_user` creates a row keyed by a synthetic email derived from the client's `session_id`, with `password_hash` left `NULL`.

Relationships: `addresses` (1:N, cascade delete). Also referenced by `cart.user_id`, `orders.user_id`, and `user_preferences.user_id` (all `ON DELETE CASCADE`).

**Known gap**: a guest's cart (tied to their shadow user's `id`) and a real account's cart (tied to a different `id` after login/signup) are separate rows with no automatic merge — see issue #84.

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

### `cart`

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `Integer` (PK, autoincrement) | |
| `user_id` | `Integer`, FK → `users.id` (`ON DELETE CASCADE`), **unique**, indexed | One cart per user — guest or real account (see the `users` table's "Known gap" note re: no cart merge on login) |
| `created_at` | `DateTime` | |
| `updated_at` | `DateTime` | See issue #87 — relies on an ORM-level Python default, not a DB `server_default` |

Relationship: `items` (1:N → `cart_items`, cascade delete).

### `cart_items`

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `Integer` (PK, autoincrement) | |
| `cart_id` | `Integer`, FK → `cart.id` (`ON DELETE CASCADE`), indexed | |
| `product_id` | `String`, FK → `products.id` (`ON DELETE CASCADE`), indexed | |
| `quantity` | `Integer`, default 1 | |
| `created_at` | `DateTime` | |

No frozen price — `cart_items` always reflects the current `products.price` at read/checkout time. Price freezing only happens at checkout, in `order_items.unit_price` below.

### `orders`

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `Integer` (PK, autoincrement) | |
| `user_id` | `Integer`, FK → `users.id` (`ON DELETE CASCADE`), indexed | |
| `address_id` | `Integer`, FK → `addresses.id` (**`ON DELETE RESTRICT`**), indexed | `RESTRICT`, not `CASCADE` like every other FK in this schema — orders are append-only financial history, so deleting an address with order history fails loudly instead of silently erasing the record of what was ordered |
| `status` | `String`, default `"paid"`, indexed | One of `pending`/`paid`/`shipped`/`delivered`/`cancelled` |
| `created_at` | `DateTime` | |

Relationships: `items` (1:N → `order_items`, cascade delete), `payment` (1:1 → `payments`, cascade delete).

### `order_items`

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `Integer` (PK, autoincrement) | |
| `order_id` | `Integer`, FK → `orders.id` (`ON DELETE CASCADE`), indexed | |
| `product_id` | `String`, FK → `products.id` (**`ON DELETE RESTRICT`**) | Same `RESTRICT` reasoning as `orders.address_id` — a product with order history can't be hard-deleted |
| `quantity` | `Integer` | |
| `unit_price` | `Float` | Frozen at checkout time — never recomputed after, unlike `cart_items` |

### `payments`

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `Integer` (PK, autoincrement) | |
| `order_id` | `Integer`, FK → `orders.id` (`ON DELETE CASCADE`), **unique**, indexed | One payment per order |
| `amount` | `Float` | |
| `status` | `String`, default `"succeeded"` | **Mocked** — there is no real payment provider integration; every checkout always succeeds. See issue #68 for the gap this leaves (no balance/payment-failure handling). |
| `provider_reference` | `String` | Fake reference string, e.g. `f"mock_{uuid4().hex}"` |
| `created_at` | `DateTime` | |

### `user_preferences`

| Field | Type | Notes |
| --- | --- | --- |
| `user_id` | `Integer`, FK → `users.id` (`ON DELETE CASCADE`), **primary key** | One row per user — no separate autoincrement id, same one-per-user shape as `cart.user_id` |
| `preferences` | `Text` | Freeform LLM-generated text summarizing durable customer preferences (e.g. preferred brands, price sensitivity) — not structured key-value facts (#54) |
| `updated_at` | `DateTime` | See issue #87 — same `server_default` gap as `cart.updated_at` |

Long-term memory (#54): populated/updated by `conversations.py`'s `maybe_summarise()`, piggybacking on the same LLM call already made for conversation summarization, only for **authenticated** users (`password_hash` is set) — guest/shadow users never get a row here. Read on every chat turn for a logged-in user and injected into `chat_history` as a `SystemMessage`, so it persists across separate conversations/sessions, unlike the Redis-only conversation summary above.

### Dropped from scope: `product_variants`

Considered (product_id FK, sku, attributes, price, stock) but dropped — confirmed the source dataset has no real variant grouping (no products share a `parent_asin`/variant relationship). Would have meant inert schema seeded with fabricated pass-through rows and zero real consumers. See issue #32 for full reasoning; revisit only if real variant data or a concrete need shows up.

## Config reference

All TTLs, thresholds, and connection URLs above are defined once in `backend/app/config.py` (`Settings`, loaded via `pydantic-settings` from environment variables / `.env`) — treat that file as the source of truth if any of the defaults listed here change.
