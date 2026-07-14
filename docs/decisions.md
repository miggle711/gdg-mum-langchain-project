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
