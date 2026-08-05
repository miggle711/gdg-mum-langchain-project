# backend/app/graph.py
import json
import logging
from typing import Any, Dict, Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langfuse import get_client
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

Intent = Literal["product_search", "product_details", "cart_action", "clarify", "fallback", "unsafe"]
ALLOWED_INTENTS = {"product_search", "product_details", "cart_action", "clarify", "fallback", "unsafe"}

CartActionType = Literal["add", "remove", "update_quantity"]


class GraphState(TypedDict, total=False):
    input: str
    chat_history: list[BaseMessage]
    session_id: str
    intent: Intent
    product_reference: str
    quantity: int
    cart_action_type: CartActionType
    response: str
    error: bool


_intent_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.google_api_key,
    temperature=0,
)


class IntentEntityExtraction(BaseModel):
    intent: Literal[
        "product_search", "product_details", "cart_action", "clarify", "fallback", "unsafe"
    ] = Field(description="The best matching intent label for the user's message.")
    product_reference: str | None = Field(
        default=None,
        description=(
            "A short natural-language description of the product being discussed or acted on "
            "(e.g. 'the blue waterproof jacket'), resolved using chat history for follow-ups "
            "like 'that one' or 'the first one'. Only set for product_details and cart_action "
            "intents; otherwise null."
        ),
    )
    quantity: int | None = Field(
        default=None,
        description="The quantity mentioned for a cart action, if any. Only set for cart_action intent.",
    )
    cart_action_type: CartActionType | None = Field(
        default=None,
        description=(
            "Which cart operation the user wants: 'add' to add items, 'remove' to remove an "
            "item entirely, or 'update_quantity' to change an existing item's quantity to a "
            "new total. Only set for cart_action intent."
        ),
    )


_INTENT_ENTITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Classify the user's message into exactly one ecommerce routing intent, and extract minimal entities.

Intents:
- product_search: browsing/discovery - searching, recommendations, comparisons, filtering by category/price/rating.
- product_details: asking about one specific, already-identified product (attributes, price, availability, follow-up questions about "it").
- cart_action: adding, removing, or changing the quantity of an item in the cart.
- clarify: unclear messages that cannot be routed using the current message or chat history.
- fallback: greetings, thanks, farewells, casual conversation, or anything else unrelated to shopping.
- unsafe: self-harm, violence, abuse, threats, illegal wrongdoing, or safety-sensitive content.

Rules:
- Use chat history to resolve follow-ups like "what about one in blue?" or "add that one".
- For cart_action, also set cart_action_type ('add', 'remove', or 'update_quantity'), product_reference, and quantity if mentioned.
- For product_details and cart_action, set product_reference to a short description of the product, using chat history to resolve vague references.
- Prefer unsafe whenever safety risk is present, regardless of any other content in the message.
- Use clarify only when no other intent clearly fits.

Examples:
- "Show me waterproof jackets under $100" -> product_search
- "What about one in blue?" after a search -> product_search
- "Tell me more about that first one" -> product_details, product_reference="the first jacket shown"
- "Add 2 of those to my cart" after viewing a product -> cart_action, cart_action_type="add", product_reference="that product", quantity=2
- "Remove the blue jacket from my cart" -> cart_action, cart_action_type="remove", product_reference="the blue jacket"
- "Actually make it 3" after adding an item -> cart_action, cart_action_type="update_quantity", product_reference="that item", quantity=3
- "Hey, how are you?" -> fallback
- "Thanks, that's all" -> fallback
- "I want to hurt someone" -> unsafe
- "asdf qwerty" -> clarify

Return only the requested fields.""",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
    ]
)

_intent_entity_extractor = _INTENT_ENTITY_PROMPT | _intent_llm.with_structured_output(IntentEntityExtraction)


async def resolve_product_reference(reference: str) -> str | None:
    """Resolve a natural-language product reference (e.g. "the blue jacket")
    to a catalog product_id, via semantic_search_impl's top match.

    Shared by the upcoming CA (cart action) and RD/PD (product details)
    nodes (#57) so reference resolution isn't duplicated between them.
    Not wired into any node yet.

    tools.py is imported lazily here, not at module level, so importing
    graph.py itself doesn't require tools.py's dependencies (e.g.
    elasticsearch) to be installed — same reasoning as _invoke_product_agent's
    lazy `from app.agent import agent_executor` below.
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools import semantic_search_impl

    raw = await semantic_search_impl(reference, limit=1)
    results = json.loads(raw).get("results") or []
    return results[0].get("id") if results else None


langfuse_client = get_client()

def _start_graph_span(name: str, state: GraphState):
    return langfuse_client.start_as_current_span(
        name=name,
        input={
            "input": state.get("input", ""),
            "chat_history_length": len(state.get("chat_history", [])),
        },
        metadata={"component": "langgraph"},
    )

async def extract_intent_and_entities(state: GraphState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    with _start_graph_span("graph.extract_intent_and_entities", state) as span:
        try:
            result = await _intent_entity_extractor.ainvoke(
                {
                    "input": state.get("input", ""),
                    "chat_history": state.get("chat_history", []),
                },
                config=config,
            )
        except Exception as exc:
            span.update(
                level="ERROR",
                status_message=str(exc),
                output={"intent": "clarify", "error": True},
            )
            logger.exception("Intent+entity extraction failed")
            # Routes to clarify_node like a genuine ambiguous-message case,
            # but error=True lets callers (app/routes/chat.py) tell a real
            # failure apart from an ordinary clarify response instead of the
            # two looking identical to the caller (#77).
            return {"intent": "clarify", "error": True}

        intent = getattr(result, "intent", "clarify")
        if intent not in ALLOWED_INTENTS:
            intent = "clarify"

        output: Dict[str, Any] = {"intent": intent}
        if intent in ("product_details", "cart_action") and result.product_reference:
            output["product_reference"] = result.product_reference
        if intent == "cart_action":
            if result.quantity is not None:
                output["quantity"] = result.quantity
            if result.cart_action_type:
                output["cart_action_type"] = result.cart_action_type

        span.update(output=output)
        return output


async def _invoke_product_agent(state: GraphState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    from app.agent import agent_executor
    return await agent_executor.invoke(
        {
            "input": state.get("input", ""),
            "chat_history": state.get("chat_history", []),
        },
        config=config,
        session_id=state.get("session_id"),
    )


async def product_node(state: GraphState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    with _start_graph_span("graph.product_node", state) as span:
        result = await _invoke_product_agent(state, config=config)
        response = result.get("output") or "I apologize, but I'm having trouble generating a response at the moment."
        span.update(output={
            "response": response,
            "tool_calls": result.get("tool_calls", []),
            })
        return {"response": response}


def small_talk_node(state: GraphState) -> Dict[str, Any]:
    with _start_graph_span("graph.small_talk_node", state) as span:
        response = "response from graph - small talk node placeholder"
        span.update(output={"response": response})
        return {"response": response}


def sensitive_node(state: GraphState) -> Dict[str, Any]:
    with _start_graph_span("graph.sensitive_node", state) as span:
        response = "response from graph - sensitive topic node placeholder"
        span.update(output={"response": response})
        return {"response": response}


async def clarify_node(state: GraphState) -> Dict[str, Any]:
    with _start_graph_span("graph.clarify_node", state) as span:
        response = "response from graph - clarify node placeholder"
        span.update(output={"response": response})
        return {"response": response}


def route_from_intent(state: GraphState) -> str:
    intent = state.get("intent", "clarify")

    # TEMPORARY (#57): PS/PD/RD/VR/GR and CA/CV/EC/CG don't exist yet, so
    # product_search/product_details still go through the old free-form
    # product_node, and cart_action has nowhere real to go yet — clarify_node
    # is a safe placeholder until the cart-action nodes are built.
    if intent in ("product_search", "product_details"):
        route = "product_node"
    elif intent == "cart_action":
        route = "clarify_node"
    elif intent == "unsafe":
        route = "sensitive_node"
    elif intent == "fallback":
        route = "small_talk_node"
    else:
        route = "clarify_node"

    with langfuse_client.start_as_current_span(
        name="graph.route_from_intent",
        input={"intent": intent},
        output={"route": route},
        metadata={"component": "langgraph"},
    ):
        pass

    return route


def build_chat_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("extract_intent_and_entities", extract_intent_and_entities)
    workflow.add_node("product_node", product_node)
    workflow.add_node("small_talk_node", small_talk_node)
    workflow.add_node("sensitive_node", sensitive_node)
    workflow.add_node("clarify_node", clarify_node)

    workflow.add_edge(START, "extract_intent_and_entities")
    workflow.add_conditional_edges(
        "extract_intent_and_entities",
        route_from_intent,
    )

    workflow.add_edge("product_node", END)
    workflow.add_edge("small_talk_node", END)
    workflow.add_edge("sensitive_node", END)
    workflow.add_edge("clarify_node", END)

    return workflow.compile()


chat_graph = build_chat_graph()
