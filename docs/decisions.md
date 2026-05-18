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