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
    resolved_product_id: str
    cart_action_valid: bool
    cart_action_result: str
    retrieved_data: str
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


# --- Grounded response generation (GR, #57) ---
# Its own LLM, at a higher temperature than the intent/entity extractor,
# matching app/agent.py's _llm (0.7) — this one writes conversational replies,
# not structured classifications, so some variation is desirable here.
_response_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.google_api_key,
    temperature=0.7,
)

_GROUNDED_RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful ecommerce customer service assistant. Answer the customer's message using ONLY the product data provided below - do not invent details that aren't present in it.

Product data (JSON):
{retrieved_data}

Guidelines:
- Be polite, professional, and concise.
- When describing products, include price, rating, and number of reviews if available.
- If multiple products are present, briefly summarize the most relevant ones rather than listing every field for every item.
- Do not mention "JSON" or that you were given data - just answer naturally, as if you already knew this.""",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
    ]
)

_grounded_response_chain = _GROUNDED_RESPONSE_PROMPT | _response_llm


def _ensure_backend_on_path() -> None:
    """tools.py/cart_tools.py live at backend/*.py, outside the app package —
    same sys.path hack app/agent.py uses at module level, but applied lazily
    (see resolve_product_reference/retrieve_data) and guarded so repeated
    calls in a long-running server don't keep appending duplicate entries.
    """
    import os
    import sys
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)


async def resolve_product_reference(reference: str) -> str | None:
    """Resolve a natural-language product reference (e.g. "the blue jacket")
    to a catalog product_id, via semantic_search_impl's top match.

    Shared by the CA (cart action) and RD/PD (product details) nodes (#57)
    so reference resolution isn't duplicated between them.

    tools.py is imported lazily here, not at module level, so importing
    graph.py itself doesn't require tools.py's dependencies (e.g.
    elasticsearch) to be installed.
    """
    _ensure_backend_on_path()
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


async def product_search_node(state: GraphState) -> Dict[str, Any]:
    """PS (#57): thin pass-through so product-search turns get their own
    trace span before retrieve_data runs the actual lookup.
    """
    with _start_graph_span("graph.product_search_node", state):
        return {}


async def product_details_node(state: GraphState) -> Dict[str, Any]:
    """PD (#57): thin pass-through, mirrors product_search_node."""
    with _start_graph_span("graph.product_details_node", state):
        return {}


async def retrieve_data(state: GraphState) -> Dict[str, Any]:
    """RD (#57): deterministic retrieval, no LLM tool-calling loop (decision
    #3). product_search hands the raw user text straight to
    semantic_search_impl, which parses natural-language queries itself
    (decision #2 — IE doesn't pre-extract filters). product_details instead
    resolves IE's extracted product_reference to a concrete id first, then
    does an exact lookup. Both paths normalize their output to the same
    {"results": [...], "count": n} shape so VR/GR don't need to know which
    path was taken.
    """
    _ensure_backend_on_path()
    from tools import semantic_search_impl
    from cart_tools import get_product_impl

    with _start_graph_span("graph.retrieve_data", state) as span:
        if state.get("intent") == "product_details":
            product_id = None
            reference = state.get("product_reference")
            if reference:
                product_id = await resolve_product_reference(reference)

            if product_id:
                product = json.loads(await get_product_impl(product_id))
                retrieved = (
                    {"results": [], "count": 0}
                    if "error" in product
                    else {"results": [product], "count": 1}
                )
            else:
                retrieved = {"results": [], "count": 0}
        else:
            parsed = json.loads(await semantic_search_impl(state.get("input", ""), limit=5))
            retrieved = {"results": parsed.get("results", []), "count": parsed.get("count", 0)}

        span.update(output={"result_count": retrieved["count"]})
        return {"retrieved_data": json.dumps(retrieved)}


async def generate_grounded_response(state: GraphState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """GR (#57): the one genuinely new LLM call in this redesign - turns
    RD's retrieved_data into a natural-language reply grounded in the
    actual product data, replacing the old free-form agent loop's job of
    describing results in prose.
    """
    with _start_graph_span("graph.generate_grounded_response", state) as span:
        fallback = "I apologize, but I'm having trouble generating a response at the moment."
        try:
            result = await _grounded_response_chain.ainvoke(
                {
                    "input": state.get("input", ""),
                    "chat_history": state.get("chat_history", []),
                    "retrieved_data": state.get("retrieved_data", "{}"),
                },
                config=config,
            )
            response = result.content or fallback
        except Exception as exc:
            span.update(level="ERROR", status_message=str(exc), output={"response": None})
            logger.exception("Grounded response generation failed")
            return {"response": fallback}

        span.update(output={"response": response})
        return {"response": response}


def validate_results(state: GraphState) -> Dict[str, Any]:
    """VR (#57): checks whether RD found anything, for route_from_validation
    to act on. Doesn't change state itself — exists as its own node (rather
    than folding the check into route_from_validation) so it gets its own
    trace span, matching the diagram.
    """
    with _start_graph_span("graph.validate_results", state) as span:
        retrieved = json.loads(state.get("retrieved_data") or "{}")
        span.update(output={"count": retrieved.get("count", 0)})
        return {}


def route_from_validation(state: GraphState) -> str:
    retrieved = json.loads(state.get("retrieved_data") or "{}")
    has_results = bool(retrieved.get("results"))
    route = "generate_grounded_response" if has_results else "clarify_node"

    with langfuse_client.start_as_current_span(
        name="graph.route_from_validation",
        input={"has_results": has_results},
        output={"route": route},
        metadata={"component": "langgraph"},
    ):
        pass

    return route


# --- Cart-action path (CA/CV/EC/CG, #57) ---
# CA and CV are purely deterministic — no LLM reasoning inside either
# (decision #4). EC/CG follow in the next step.

MAX_CART_ACTION_QUANTITY = 20


async def interpret_cart_action(state: GraphState) -> Dict[str, Any]:
    """CA: resolves IE's extracted product_reference to a concrete catalog
    id via resolve_product_reference. quantity/cart_action_type pass
    through from IE unchanged — CV (next) validates them.
    """
    with _start_graph_span("graph.interpret_cart_action", state) as span:
        product_id = None
        reference = state.get("product_reference")
        if reference:
            product_id = await resolve_product_reference(reference)

        span.update(output={"resolved_product_id": product_id})
        return {"resolved_product_id": product_id} if product_id else {}


async def validate_cart_action(state: GraphState) -> Dict[str, Any]:
    """CV: the guardrail before any mutation — product exists, the action
    type is one of add/remove/update_quantity, and quantity is a sane
    positive number within MAX_CART_ACTION_QUANTITY.

    Since EC (next step) does no LLM reasoning either, this is the only
    check standing between a possibly-manipulated IE/CA output and a real
    cart write. Note this guards against unwanted mutations to the
    requesting user's *own* cart, not cross-user access — session_id is
    always trusted/HTTP-sourced, never LLM-derived (same as EC), so
    cross-session mutation isn't possible regardless of what IE/CA produce.
    """
    _ensure_backend_on_path()
    from cart_tools import get_product_impl

    with _start_graph_span("graph.validate_cart_action", state) as span:
        product_id = state.get("resolved_product_id")
        action_type = state.get("cart_action_type")
        quantity = state.get("quantity")

        product_exists = False
        if product_id:
            product = json.loads(await get_product_impl(product_id))
            product_exists = "error" not in product

        if action_type == "remove":
            quantity_valid = True
        elif quantity is None:
            # add_to_cart_impl defaults quantity to 1 when omitted;
            # update_quantity_impl has no sensible default — it sets an
            # absolute new total, so an explicit quantity is required.
            quantity = 1 if action_type == "add" else None
            quantity_valid = quantity is not None
        else:
            quantity_valid = 0 < quantity <= MAX_CART_ACTION_QUANTITY

        valid = (
            bool(product_id)
            and product_exists
            and action_type in ("add", "remove", "update_quantity")
            and quantity_valid
        )

        span.update(output={
            "valid": valid,
            "product_exists": product_exists,
            "quantity_valid": quantity_valid,
        })
        output: Dict[str, Any] = {"cart_action_valid": valid}
        if quantity is not None:
            output["quantity"] = quantity
        return output


def route_from_cart_validation(state: GraphState) -> str:
    valid = state.get("cart_action_valid", False)
    route = "execute_cart_action" if valid else "clarify_node"

    with langfuse_client.start_as_current_span(
        name="graph.route_from_cart_validation",
        input={"valid": valid},
        output={"route": route},
        metadata={"component": "langgraph"},
    ):
        pass

    return route


async def execute_cart_action(state: GraphState) -> Dict[str, Any]:
    """EC: deterministic dispatch straight to the cart-mutation impl
    functions — no StructuredTool, no LLM tool-calling loop (decision
    #3/#4). session_id always comes from GraphState (trusted, HTTP-sourced),
    never from IE/CA's output — same trusted-injection pattern app/agent.py
    already uses for these same functions via the ReAct loop.

    CV (previous step) has already validated product_id/action_type/
    quantity by the time this runs, so no re-validation happens here.
    """
    _ensure_backend_on_path()
    from cart_tools import add_to_cart_impl, remove_from_cart_impl, update_quantity_impl

    with _start_graph_span("graph.execute_cart_action", state) as span:
        action_type = state.get("cart_action_type")
        product_id = state.get("resolved_product_id")
        quantity = state.get("quantity")
        session_id = state.get("session_id")

        if action_type == "add":
            raw = await add_to_cart_impl(product_id, quantity, session_id=session_id)
        elif action_type == "remove":
            raw = await remove_from_cart_impl(product_id, session_id=session_id)
        elif action_type == "update_quantity":
            raw = await update_quantity_impl(product_id, quantity, session_id=session_id)
        else:
            raw = json.dumps({"error": f"Unsupported cart action: {action_type}"})

        span.update(output={"result": raw})
        return {"cart_action_result": raw}


def generate_cart_confirmation(state: GraphState) -> Dict[str, Any]:
    """CG: deterministic, no LLM call. add_to_cart_impl/remove_from_cart_impl/
    update_quantity_impl already return a human-readable "message" field
    (e.g. "Added 2 x Blue Jacket to cart"), so reusing it directly stays
    below decision #4's "one call for GR or CG" budget rather than using it.
    """
    with _start_graph_span("graph.generate_cart_confirmation", state) as span:
        result = json.loads(state.get("cart_action_result") or "{}")
        response = result.get("message") or result.get("error") or "Sorry, I couldn't update your cart."
        span.update(output={"response": response})
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

    # TEMPORARY (#57): CA/CV/EC/CG don't exist yet, so cart_action has
    # nowhere real to go — clarify_node is a safe placeholder until the
    # cart-action nodes are built (next step).
    if intent == "product_search":
        route = "product_search_node"
    elif intent == "product_details":
        route = "product_details_node"
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
    workflow.add_node("product_search_node", product_search_node)
    workflow.add_node("product_details_node", product_details_node)
    workflow.add_node("retrieve_data", retrieve_data)
    workflow.add_node("validate_results", validate_results)
    workflow.add_node("generate_grounded_response", generate_grounded_response)
    workflow.add_node("small_talk_node", small_talk_node)
    workflow.add_node("sensitive_node", sensitive_node)
    workflow.add_node("clarify_node", clarify_node)

    workflow.add_edge(START, "extract_intent_and_entities")
    workflow.add_conditional_edges(
        "extract_intent_and_entities",
        route_from_intent,
    )

    workflow.add_edge("product_search_node", "retrieve_data")
    workflow.add_edge("product_details_node", "retrieve_data")
    workflow.add_edge("retrieve_data", "validate_results")
    workflow.add_conditional_edges(
        "validate_results",
        route_from_validation,
    )

    workflow.add_edge("generate_grounded_response", END)
    workflow.add_edge("small_talk_node", END)
    workflow.add_edge("sensitive_node", END)
    workflow.add_edge("clarify_node", END)

    return workflow.compile()


chat_graph = build_chat_graph()
