# Decisions

This document will keep track of major architectural decisions made by the dev team for future refernce.

## Transition from ConversationChain to AgentExecutor

- The current implementation as of [the agent](https://github.com/miggle711/gdg-mum-langchain-project/commit/06dd2701b6d6b3b12ebb20134b422565c8dac4a2) has used the CoversationChain 
- It is (apparently) deprecated lolz
- A basic memory wrapper

- *AgentExecutor* is a runtime class that manages the execution loop for an AI agent
- LLM can call tools and the class can execute them
- LLM automatically integrates the aoutput of the tool into the resposnse
- The cycle repeats: LLM > tool > LLM > tool...
- We will add this to support tooling for DB querying

## Memory Architecture 

- Switched from ConversationBufferMemory (string buffer) to InMemoryChatMessageHistory (message list). 
- History passed at invoke time, not bound at creation. (modern message-based pattern, better with ReAct agents)

## Transition from direct AgentExecutor routes to LangGraph routing

- The chat routes no longer call the product agent directly from `chat.py`.
- `backend/app/graph.py` now classifies each message into one of four intents:
  - `product_details`
  - `small_talk`
  - `sensitive_topic`
  - `clarify`
- LangGraph is now the outer orchestration layer for `/chat` and `/chat/stream`.
- The `product_details` branch still delegates to the existing product runtime in `backend/app/agent.py`.
- This is an intentional transitional design:
  - lower-risk migration
  - preserves the existing product tool-calling behavior
  - allows graph-native branching for non-product paths

## Langfuse tracing moved from callback-only flow to explicit request spans

- Callback-based tracing alone did not guarantee a visible root trace for every request.
- Non-product graph branches could complete with little or no downstream LangChain activity, which made traces harder to find consistently.
- `backend/app/routes/chat.py` now creates explicit request-level Langfuse spans for `/chat` and `/chat/stream`.
- `backend/app/graph.py` adds child spans around:
  - intent classification
  - branch routing
  - terminal graph nodes
- This makes each request traceable even when no product tools are called.

## Restructure to a deterministic, cart-capable graph (#57)

- Replaced the 4-intent classifier (`product_details`/`small_talk`/`sensitive_topic`/`clarify`) with a single `extract_intent_and_entities` call producing 5 intents: `product_query`, `cart_action`, `clarify`, `fallback`, `unsafe`.
- Safety folded into this same routing call rather than a dedicated node — no incident motivating a stricter posture, and a separate call would double AI cost on every message to catch a rare case.
- The old free-form `AgentExecutorAdapter` tool-calling loop in `backend/app/agent.py` (product queries previously delegated to it) was removed entirely — `retrieve_data`/`execute_cart_action` now call `tools.py`/`cart_tools.py` functions directly, deterministically, no LLM picking tools mid-conversation.
- Added a cart-action pipeline (`interpret_cart_action` → `validate_cart_action` → `execute_cart_action` → `generate_cart_confirmation`) with an explicit validation guardrail (product exists, quantity sane) before any cart mutation — zero AI calls after routing.
- `product_search`/`product_details` initially shipped as two separate intents, then merged into one (`product_query`) after the two-intent split proved to be a hard, unrecoverable upfront guess that broke on follow-up questions and mixed messages. `retrieve_data` now always searches and additionally fetches a specific product's detail record when referenced, letting the final reply-writing step decide how to answer from the user's actual phrasing.
- Added hard filter support (`category`/`price_min`/`price_max`/`rating_min`) to the search path, decided by actually measuring against `golden_dataset.json` (the team's pre-existing eval set) rather than by argument alone: unfiltered search found 0/26 anchor products on exact price/rating queries; filtering plus a rating/reviews sort override (matching `query_products_impl`'s already-validated convention, since filtered queries often have too little descriptive text for relevance ranking to work with) brought that to 26/33 (79%), against the old exact-filter tool's 27/33 (82%) on the same cases.
