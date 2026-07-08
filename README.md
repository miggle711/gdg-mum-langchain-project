# gdg-mum-langchain-project

A LangChain-powered ecommerce customer service chatbot: an Angular frontend, a FastAPI backend running a Gemini-based tool-calling agent, hybrid product search over Elasticsearch, and a Redis-backed semantic cache and conversation store. Deployed with Docker.

## Overview

The backend exposes a chat agent that can search a product catalog and answer customer questions about it. On each turn, a LangChain `AgentExecutor` (Gemini 2.5 Flash) decides whether to call one of three product tools, then responds using the results. Conversation history and summaries live in Redis so the agent stays stateful across requests without keeping anything in process memory. Every chat call is traced in Langfuse, and users can leave thumbs up/down feedback tied to that trace.

Product search: Elasticsearch runs a hybrid BM25 + kNN vector query to fetch candidates, a cross-encoder reranks them for relevance, and a Redis vector index caches results by query embedding so near-duplicate questions skip the expensive path.

## Tech Stack

**Backend:**

- FastAPI, Pydantic (`pydantic-settings` for config)
- LangChain (`AgentExecutor` + tool calling) with `langchain-google-genai` (Gemini 2.5 Flash)
- Elasticsearch: product catalog, hybrid (BM25 + kNN) search
- `sentence-transformers`: `BAAI/bge-base-en-v1.5` embeddings, `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker
- Redis Stack: conversation history/summaries, rate-limit counters, semantic search cache (vector index)
- Langfuse; tracing and user feedback scoring
- `prometheus-fastapi-instrumentator` + custom counters/histograms — metrics
- `slowapi`: per-IP rate limiting

**Frontend:**

- Angular 19+, TypeScript, RxJS Observables, Angular Material

**Deployment:**

- Docker & Docker Compose
- nginx (serves the Angular build, reverse-proxies `/api/*` to the backend)

## Architecture

1. **Frontend** calls `POST /chat/start` on load; the backend generates a `conversation_id` and returns a welcome message.
2. **Frontend** sends each user message to `POST /chat` (or `/chat/stream` for token streaming) with that `conversation_id`.
3. **Backend** loads prior messages for that conversation from Redis, summarising older turns once the history passes a configurable threshold (`conversation.py`'s `maybe_summarise`), then invokes the LangChain agent with the user's message and recent history.
4. **The agent** (Gemini 2.5 Flash + `ECOMMERCE_SYSTEM_PROMPT`) decides whether to call a tool:
   - `semantic_search` — vague/descriptive queries ("something cozy for winter"), backed by the ES hybrid search + rerank pipeline
   - `query_products` — exact filters (category, price range, rating), backed by a plain ES bool query
   - `list_categories` — so the agent uses exact category names rather than guessing
5. **Backend** saves the updated conversation back to Redis, and returns the response along with a `trace_id` for Langfuse.
6. **Frontend** can submit feedback (thumbs up/down) against that `trace_id` via `POST /feedback`.

### Search pipeline (`backend/search.py`, `backend/cache.py`)

```mermaid
flowchart TD
    Q[Query embedding] --> C{Redis semantic cache<br/>KNN cosine lookup}
    C -->|hit: distance within threshold| R1[Return cached results]
    C -->|miss| ES[Elasticsearch hybrid query<br/>BM25 multi_match boost 0.5<br/>+ kNN on embedding boost 4.0]
    ES --> CE[Cross-encoder reranks candidates<br/>query, name+description pairs]
    CE --> Cache[Cache top-N results in Redis<br/>keyed by query embedding]
    Cache --> R2[Return results]
```

Product embeddings are generated once at index time (`backend/scripts/index_products.py`); query embeddings are generated per-request in `backend/tools.py`. Both use the same BGE model with matching (but asymmetric) instruction prefixes — `"Represent this product for retrieval: ..."` for documents, `"Represent this sentence for searching relevant passages: ..."` for queries — and both normalize embeddings so cosine similarity is meaningful.

See [backend/DATABASE.md](backend/DATABASE.md) for the Elasticsearch index mapping and tool contracts, and [docs/db-models.md](docs/db-models.md) for the full data reference (ES fields + Redis key schemas).

### Conversation Flow

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant FE as Angular Frontend
    participant BE as FastAPI (/chat)
    participant R as Redis
    participant Agent as LangChain Agent<br/>(Gemini 2.5 Flash)
    participant LF as Langfuse

    U->>FE: Open chat
    FE->>BE: POST /chat/start
    BE->>FE: conversation_id + welcome message

    U->>FE: Send message
    FE->>BE: POST /chat {conversation_id, message}
    BE->>R: load_messages(conversation_id)
    R-->>BE: prior messages (+ summary if any)
    alt history over summary threshold
        BE->>Agent: summarise older turns (LLM call)
        Agent-->>BE: summary text
        BE->>R: save_summary + save_messages (trimmed)
    end

    BE->>LF: create_trace_id()
    BE->>Agent: invoke(input, chat_history, callbacks=[Langfuse handler])
    Agent->>Agent: pick a tool — semantic_search,<br/>query_products, or list_categories
    Note over Agent: tool executes against Elasticsearch/Redis —<br/>see Search Pipeline diagram above
    Agent-->>BE: final response text

    BE->>R: save_messages(conversation_id, updated history)
    BE-->>FE: {response, trace_id}
    FE-->>U: Render assistant reply

    opt user rates the reply
        U->>FE: Thumbs up / down
        FE->>BE: POST /feedback {trace_id, value}
        BE->>LF: create_score(trace_id, value)
    end
```

`/chat/stream` follows the same shape but streams agent output tokens over Server-Sent Events instead of returning one JSON payload.

## Features

- Tool-calling agent (LangChain `AgentExecutor`) over a real product catalog, not just a system-prompted chatbot
- Hybrid lexical + semantic product search with cross-encoder reranking
- Semantic response caching in Redis (near-duplicate queries skip search entirely)
- Server-side conversation history in Redis with automatic summarisation for long conversations
- Streaming responses via Server-Sent Events (`/chat/stream`)
- Langfuse tracing on every chat turn, with user feedback (thumbs up/down) tied to a trace
- Prometheus metrics (`/metrics`) — cache hit/miss counters, ES/rerank latency histograms, standard HTTP metrics
- Per-IP rate limiting (20 requests/minute on chat endpoints), backed by Redis so it's consistent across replicas
- `/health` endpoint reporting Elasticsearch and Redis connectivity
- Full Docker Compose deployment (frontend, backend, Redis Stack, Elasticsearch)

## Getting Started

### Prerequisites
- Docker Desktop
- `.env` file in project root with `GOOGLE_API_KEY` set (see [.env.example](.env.example))
- A `backend/.env` for backend-specific overrides — see [backend/.env.example](backend/.env.example) (Redis/ES URLs, cache/conversation TTLs, Langfuse keys)

### Run Locally

```bash
docker compose up --build
```

Then open `http://localhost` in your browser.

### Load sample product data

The Elasticsearch index is created empty on first startup. To populate it with a sample Amazon product catalog (~2k products across 4 categories):

```bash
docker compose exec backend python scripts/index_products.py
```

This generates BGE embeddings for each product and bulk-indexes them — see [backend/DATABASE.md](backend/DATABASE.md) for details.

### Setup Guides

- **[DOCKER_SETUP.md](DOCKER_SETUP.md)** — Docker prerequisites, environment setup, troubleshooting
- **[backend/DATABASE.md](backend/DATABASE.md)** — Elasticsearch index, search tools, indexing script
- **[docs/db-models.md](docs/db-models.md)** — full data reference (ES mapping, Redis key schemas)
- **[docs/decisions.md](docs/decisions.md)** — log of major architectural decisions and why they were made

## Key Files

- **backend/app/main.py** — FastAPI app setup: middleware, rate limiter, exception handling, startup (index/cache init)
- **backend/app/agent.py** — LangChain agent definition (system prompt, LLM, tools, Langfuse client)
- **backend/app/routes/chat.py** — chat/stream/feedback/conversation endpoints
- **backend/search.py** — Elasticsearch hybrid search, reranking, category listing
- **backend/cache.py** — Redis semantic search cache (vector index)
- **backend/conversations.py** — Redis-backed conversation history + summarisation
- **backend/tools.py** — LangChain tool definitions the agent calls (wraps `search.py`)
- **backend/scripts/index_products.py** — one-time script to load and embed sample product data
- **backend/app/config.py** — centralized settings (`pydantic-settings`), single source of truth for all tunables
- **frontend/chatbot-ui/src/app/services/chat.ts** — HTTP/SSE service layer for the chat API
- **frontend/chatbot-ui/src/app/components/chat-panel/chat-panel.ts** — Chat UI with message handling
- **docker-compose.yml** — orchestrates backend, frontend, Redis Stack, and Elasticsearch

## API Endpoints

- `POST /chat/start` — Initialize conversation, returns `conversation_id` and welcome message
- `POST /chat` — Send message, returns AI response + `trace_id`
- `POST /chat/stream` — Same as `/chat`, but streams the response token-by-token via SSE
- `GET /conversation/{id}` — Retrieve conversation history
- `DELETE /conversation/{id}` — Delete conversation
- `GET /conversations` — List all conversations
- `POST /feedback` — Submit thumbs up/down (+ optional comment) for a given `trace_id`
- `GET /health` — Elasticsearch + Redis connectivity status
- `GET /metrics` — Prometheus metrics

## Development Notes

- Conversations are stored in Redis with a TTL (`conversation_ttl_seconds`, default 24h) — they survive backend restarts but expire eventually, not "forever."
- The system prompt is hardcoded in `backend/app/agent.py` for ecommerce customer service; it is not currently configurable per-request.
- Elasticsearch and Redis indices are created automatically on backend startup if they don't already exist (`init_es_index`, `init_cache_index`) — this happens synchronously at import time, so the backend will fail to start if either service is unreachable.
- See [docs/decisions.md](docs/decisions.md) for why certain architecture choices were made (e.g. ConversationChain → AgentExecutor, LangSmith → Langfuse).
