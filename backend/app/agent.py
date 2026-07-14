import os
import sys
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse, get_client

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import PRODUCT_TOOLS
from app.config import settings

ECOMMERCE_SYSTEM_PROMPT = """You are a helpful and professional ecommerce customer service assistant for our online store.

Your responsibilities:
- Help customers find products by answering questions about our catalog
- Use the available tools to search and filter products when customers ask
- Provide information about product specifications, pricing, and availability

Tool usage guidelines:
- Use semantic_search for vague or descriptive queries like "something cozy for winter" or "gift for a fitness lover"
- Use query_products for exact filters like "electronics under $50" or "books with rating above 4.5"
- When filtering by category with query_products, call list_categories first to get exact category names
- You can combine both tools — semantic_search to find relevant products, then describe them with price/rating details

Response guidelines:
- Always be polite, professional, and empathetic
- When describing products, include price, rating, and number of reviews
- If you don't know something about our specific products or policies, suggest they contact support"""

MAX_TOOL_ITERATIONS = 5

_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.google_api_key,
    temperature=0.7,
)
_tool_enabled_llm = _llm.bind_tools(PRODUCT_TOOLS)
_tool_map = {tool.name: tool for tool in PRODUCT_TOOLS}


def _build_messages(payload: dict[str, Any]) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=ECOMMERCE_SYSTEM_PROMPT)]
    messages.extend(payload.get("chat_history", []))

    user_input = payload.get("input", "")
    if user_input:
        messages.append(HumanMessage(content=user_input))

    return messages


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content or "")


def _run_product_agent(payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, str]:
    messages = _build_messages(payload)
    last_ai_message: AIMessage | None = None

    for _ in range(MAX_TOOL_ITERATIONS):
        ai_message = _tool_enabled_llm.invoke(messages, config=config)
        last_ai_message = ai_message
        messages.append(ai_message)

        if not ai_message.tool_calls:
            return {"output": _message_text(ai_message)}

        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool = _tool_map.get(tool_name)

            if tool is None:
                tool_result = f"Tool '{tool_name}' is not available."
            else:
                tool_result = tool.invoke(tool_call.get("args", {}), config=config)

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

    return {"output": _message_text(last_ai_message) if last_ai_message else ""}


class AgentExecutorAdapter:
    def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, str]:
        return _run_product_agent(payload, config=config)

    async def astream(self, payload: dict[str, Any], config: dict[str, Any] | None = None):
        result = _run_product_agent(payload, config=config)
        if result["output"]:
            yield {"output": result["output"]}


agent_executor = AgentExecutorAdapter()

Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    base_url=settings.langfuse_base_url,
)

langfuse_client = get_client()
