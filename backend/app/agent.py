from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse, get_client

from app.config import settings

# This file used to also hold a free-form ReAct tool-calling loop
# (AgentExecutorAdapter/_run_product_agent) that app/graph.py's product_node
# delegated to. #57 replaced that whole path with graph.py's deterministic
# retrieve_data/execute_cart_action nodes calling tools.py/cart_tools.py
# directly, so that loop was removed (confirmed via grep: nothing outside
# this file ever imported agent_executor). Only _llm and langfuse_client
# below are still used, by app/routes/chat.py.

_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.google_api_key,
    temperature=0.7,
)

Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    base_url=settings.langfuse_base_url,
)

langfuse_client = get_client()
