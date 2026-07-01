# Ecommerce RAG Agent: Architecture & Roadmap

## Current Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| API | FastAPI + LangChain AgentExecutor | Request handling, tool orchestration |
| LLM | Gemini 2.5 Flash | Natural language understanding |
| Search | Elasticsearch 8 | Hybrid BM25 + dense vector search |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Re-scores top 20 ES candidates for precision |
| Embeddings | BAAI/bge-base-en-v1.5 (768 dims) | Query and product vectorisation |
| Cache | Redis Stack | Conversation sessions + HNSW vector search cache |
| Frontend | Angular + Nginx | Chat UI |
| Infra | Docker Compose | Local orchestration |

---

## Architecture Diagram

```
User
 │
 ▼
FastAPI (/chat)
 │  ├─ Request ID middleware (uuid4 → X-Request-ID header)
 │  ├─ Rate limiter (slowapi, 20 req/min/IP, Redis-backed)
 │  └─ Global error handler (clean JSON 500, no stack trace leak)
 │
 ├─ Redis ──────────────── load/save conversation history (TTL 24h)
 │
 ▼
LangChain AgentExecutor (Gemini 2.5 Flash)
 │
 ├─ semantic_search tool
 │    │
 │    ├─ Redis Stack HNSW cache ── KNN vector search (threshold 0.92, TTL 6h)
 │    │    └─ HIT → return cached results instantly (~3ms)
 │    │
 │    └─ MISS →
 │         ├─ BGE bi-encoder → query embedding
 │         ├─ Elasticsearch hybrid query (BM25 + knn, top 20 candidates)
 │         ├─ Cross-encoder reranker → rescore 20 pairs, return top 5
 │         └─ Store in Redis Stack HNSW cache
 │
 └─ query_products tool
      └─ Elasticsearch → structured filters (price, rating, category)
```

---

## What Has Been Built

### Step 1 — LLM Singleton + Redis Session Persistence ✅
- Shared `AgentExecutor` created once at startup, reused across all requests
- Conversation history stored in Redis per `conversation_id` with 24h TTL
- Connection pool (`max_connections=20`) for Redis

### Step 2 — Modular FastAPI Structure ✅
- `app/routes/`, `app/models.py`, `app/agent.py`, `app/config.py`, `app/limiter.py`
- `APIRouter` for clean separation of concerns

### Step 3 — Elasticsearch Hybrid Search + Cross-Encoder Reranking ✅
- 2000 real Amazon products across 4 categories (Sports, Electronics, Home, Toys)
- Hybrid query: BM25 (`boost=0.5`) + knn dense vector (`boost=4.0`)
- Cross-encoder reranks top 20 ES candidates, returns top 5
- `query_products` and `get_categories` backed by ES (Postgres removed entirely)

### Step 4 — Redis Stack Semantic Cache ✅
- HNSW vector index over cached query embeddings (`FT.CREATE`, `M=16`, `EF_CONSTRUCTION=200`)
- Cache write: `HSET` with binary `FLOAT32` embedding + JSON results
- Cache read: single `FT.SEARCH KNN 1` query — O(log n), no Python scan loop
- Threshold 0.92 catches paraphrases ("yoga mat" → "best yoga mat for home")
- Performance: 2600ms cold → 3ms exact hit → ~400ms semantic paraphrase hit

### Step 5 — Production Hardening ✅
- **Pydantic BaseSettings** — fail-fast config validation at startup
- **Health check** — `GET /health` pings ES and Redis, returns per-service status
- **Request ID middleware** — uuid4 stamped on every request, echoed in `X-Request-ID` header
- **Global error handler** — catches unhandled exceptions, logs with request ID, returns clean JSON 500
- **Rate limiting** — `slowapi` 20 req/min/IP on `/chat`, Redis-backed counter shared across workers
- **PostgreSQL connection pooling** — `ThreadedConnectionPool` (removed when Postgres was dropped)

---

## Upcoming: AI Depth

### 6 — Conversation Summarisation

**Problem:** The full message history is appended to every Gemini request. A 30-turn conversation sends ~15k tokens of context on every call, inflating cost and latency linearly with conversation length.

**Approach:** Sliding window with LLM-generated summary. Keep the last N turns verbatim; summarise everything older into a single system message.

```
[system: "Summary of earlier conversation: the user was looking for
  running shoes under $80, found a Nike pair (rated 4.3), and then
  asked about waterproof options..."]
[human: last 6 messages]
[ai: last 6 messages]
[human: current message]
```

**Implementation plan:**
1. Add a `summarise_history(messages, llm)` function that calls Gemini with a summarisation prompt when `len(messages) > SUMMARY_THRESHOLD` (e.g. 20 messages / 10 turns)
2. Store the summary string alongside the message list in Redis: `conversation:{id}:summary`
3. In `chat.py`, load both and prepend the summary as a system message before invoking the agent
4. On each turn, check if the rolling window has grown past the threshold and re-summarise

**Tradeoffs:**
- Summarisation adds one extra LLM call when the threshold is crossed (~500ms)
- Some detail is lost in the summary — acceptable for a shopping assistant, not for a legal/medical use case
- The threshold is tunable; start at 20 messages and adjust based on observed token usage

**Effort:** ~3 hours. **Impact:** Keeps token cost flat regardless of conversation length.

---

### 7 — Product Reindex Pipeline

**Problem:** Running `python scripts/index_products.py` manually is the only way to update the product catalog. Any new products, price changes, or category additions require a developer action and a full re-index (wipes and recreates the ES index).

**Two approaches:**

#### Option A — Scheduled reindex (cron)
A nightly job re-indexes all products from the source dataset. Simple, no moving parts, but data is always up to 24h stale.

```
# docker-compose addition
indexer:
  build: ./backend
  command: python scripts/index_products.py
  depends_on:
    elasticsearch:
      condition: service_healthy
  profiles: ["indexer"]  # only runs when explicitly invoked
```

Trigger: `docker compose --profile indexer run indexer`
Or add to a CI cron job / cloud scheduler.

#### Option B — Webhook-triggered delta index (event-driven)
A `POST /admin/reindex` endpoint accepts a product payload and upserts it into ES without wiping the index. Suitable when products are managed via a CMS or PIM system.

```python
@router.post("/admin/reindex")
async def reindex_product(product: ProductPayload, api_key: str = Header(...)):
    if api_key != settings.admin_api_key:
        raise HTTPException(403)
    embedding = embed(product.name + ". " + product.description)
    es_bulk_index([{...product, "embedding": embedding}])
    return {"indexed": product.id}
```

**Recommendation:** Implement Option A first (low complexity), add Option B when there's an actual product management workflow to hook into.

**Effort:** Option A ~1 hour, Option B ~3 hours. **Impact:** Removes the manual step from catalog updates.

---

### 8 — Streaming Responses

**Problem:** `/chat` waits for the full Gemini response before sending anything to the client. For longer answers (product comparisons, detailed recommendations) this means 3–8 seconds of silence before any text appears.

**Approach:** FastAPI `StreamingResponse` + LangChain's streaming callbacks.

```python
from fastapi.responses import StreamingResponse
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

@router.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest):
    async def generate():
        async for chunk in agent_executor.astream({"input": body.message, ...}):
            if "output" in chunk:
                yield f"data: {json.dumps({'text': chunk['output']})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

The frontend switches from a single `POST /chat` call to an `EventSource` / `fetch` with `ReadableStream` to render tokens as they arrive.

**Tradeoffs:**
- Streaming and tool calls (agent reasoning steps) interact awkwardly — the agent needs to finish tool calls before streaming the final answer, so the first few seconds are still silent during tool use
- The existing `/chat` endpoint stays as-is for clients that don't support streaming
- Redis session save still happens after the full response is assembled

**Effort:** ~4 hours (backend ~2h, frontend ~2h). **Impact:** Perceived latency drops dramatically even if wall-clock time is the same.

---

### 9 — Observability (LangSmith + Prometheus)

**Problem:** There's no visibility into what the agent is doing, how often the cache is hit, or where latency is coming from. `LANGCHAIN_TRACING_V2` is wired but disabled.

#### LangSmith (already wired)
Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY=<key>` in `.env`. Every agent run — including tool calls, LLM inputs/outputs, and intermediate reasoning steps — appears in the LangSmith dashboard with latency breakdowns. Zero code change required.

#### Prometheus metrics (new)
Add `prometheus-fastapi-instrumentator` to expose a `/metrics` endpoint:

```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

Then add custom counters for the things Prometheus doesn't see automatically:

```python
from prometheus_client import Counter, Histogram

cache_hits = Counter("search_cache_hits_total", "Redis semantic cache hits")
cache_misses = Counter("search_cache_misses_total", "Redis semantic cache misses")
rerank_latency = Histogram("rerank_duration_seconds", "Cross-encoder reranking latency")
```

Instrument `get_cached_search` and `semantic_search` to increment these. Scrape with a local Prometheus + Grafana stack (one `docker-compose.yml` addition each).

**Effort:** LangSmith ~15 min (config only). Prometheus ~2 hours. **Impact:** Makes performance regressions and cache effectiveness visible.

---

## Deployment Plan

To be documented once the above steps are complete. Target platform TBD (Render, Fly.io, GCP Cloud Run).

Key prerequisites before deploying:
- Authentication on `/chat` (API key or JWT)
- HTTPS termination
- Secrets management (not `.env` file)
- Conversation summarisation (to keep token costs predictable)
