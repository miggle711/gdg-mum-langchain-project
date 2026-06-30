import os
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import PRODUCT_TOOLS

ECOMMERCE_SYSTEM_PROMPT = """You are a helpful and professional ecommerce customer service assistant for our online store.

Your responsibilities:
- Help customers find products by answering questions about our catalog
- Use the available tools to search and filter products when customers ask
- Provide information about product specifications, pricing, and availability

Guidelines:
- Always be polite, professional, and empathetic
- When describing products, include price, rating, and number of reviews
- If you don't know something about our specific products or policies, suggest they contact support"""

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Shared singletons — created once at startup and reused across all conversations.
# The LLM client, prompt, and AgentExecutor are stateless between calls;
# per-conversation state lives entirely in the chat_history we pass in each invocation.
_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", ECOMMERCE_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent_executor = AgentExecutor(
    agent=create_tool_calling_agent(_llm, PRODUCT_TOOLS, _prompt),
    tools=PRODUCT_TOOLS,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
)
