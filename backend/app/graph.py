# backend/app/graph.py
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

Intent = Literal["product_details", "small_talk", "sensitive_topic", "clarify"]
ALLOWED_INTENTS = {"product_details", "small_talk", "sensitive_topic", "clarify"}


class GraphState(TypedDict, total=False):
    input: str
    chat_history: list[BaseMessage]
    session_id: str
    intent: Intent
    response: str


class IntentClassification(BaseModel):
    intent: Intent = Field(
        description="The best matching intent label for the user's message.",
    )


_INTENT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Classify the user's message into exactly one ecommerce routing intent.

Intents:
- product_details: shopping intent, including product search, recommendations, comparisons, prices, availability, product attributes, or follow-up product questions.
- small_talk: greetings, thanks, farewells, or casual conversation unrelated to shopping.
- sensitive_topic: self-harm, violence, abuse, threats, illegal wrongdoing, or safety-sensitive content.
- clarify: unclear messages that cannot be routed using the current message or chat history.

Rules:
- Use chat history to resolve follow-ups like "what about one in blue?"
- Prefer product_details when the user is asking about buying, comparing, finding, or choosing products.
- Prefer sensitive_topic whenever safety risk is present.
- Use clarify only when no other intent clearly fits.

Examples:
- "Show me waterproof jackets under $100" -> product_details
- "What about one in blue?" after discussing jackets -> product_details
- "Can you compare laptops for college?" -> product_details
- "Hey, how are you?" -> small_talk
- "Thanks, that's all" -> small_talk
- "I want to hurt someone" -> sensitive_topic
- "asdf qwerty" -> clarify

Return only the best intent label.""",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
    ]
)

_intent_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.google_api_key,
    temperature=0,
)

_intent_classifier = _INTENT_CLASSIFIER_PROMPT | _intent_llm.with_structured_output(IntentClassification)

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

async def classify_intent(state: GraphState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    with _start_graph_span("graph.classify_intent", state) as span:
        try:
            result = await _intent_classifier.ainvoke(
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
                output={"intent": "clarify"},
            )
            logger.exception("Intent classification failed; falling back to clarify")
            return {"intent": "clarify"}

        intent = getattr(result, "intent", "clarify")
        if intent not in ALLOWED_INTENTS:
            intent = "clarify"

        span.update(output={"intent": intent})
        return {"intent": intent}


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

    if intent == "product_details":
        route = "product_node"
    elif intent == "small_talk":
        route = "small_talk_node"
    elif intent == "sensitive_topic":
        route = "sensitive_node"
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

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("product_node", product_node)
    workflow.add_node("small_talk_node", small_talk_node)
    workflow.add_node("sensitive_node", sensitive_node)
    workflow.add_node("clarify_node", clarify_node)

    workflow.add_edge(START, "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_from_intent,
    )

    workflow.add_edge("product_node", END)
    workflow.add_edge("small_talk_node", END)
    workflow.add_edge("sensitive_node", END)
    workflow.add_edge("clarify_node", END)

    return workflow.compile()


chat_graph = build_chat_graph()
