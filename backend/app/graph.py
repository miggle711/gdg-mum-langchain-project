# backend/app/graph.py
from typing import Any, Dict, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph

Intent = Literal["product_details", "small_talk", "sensitive_topic", "clarify"]


class GraphState(TypedDict, total=False):
    input: str
    chat_history: list[BaseMessage]
    intent: Intent
    response: str


def classify_intent(state: GraphState) -> Dict[str, Any]:
    text = state.get("input", "").lower()

    if any(word in text for word in ["sensitive"]):
        return {"intent": "sensitive_topic"}

    if any(word in text for word in ["small talk"]):
        return {"intent": "small_talk"}

    if any(word in text for word in ["search product"]):
        return {"intent": "product_details"}

    return {"intent": "clarify"}


def product_node(state: GraphState) -> Dict[str, Any]:
    return {"response": "product node placeholder"}


def small_talk_node(state: GraphState) -> Dict[str, Any]:
    return {"response": "small talk node placeholder"}


def sensitive_node(state: GraphState) -> Dict[str, Any]:
    return {"response": "sensitive topic node placeholder"}


def clarify_node(state: GraphState) -> Dict[str, Any]:
    return {"response": "I’m not sure which path fits best. Are you asking about a product, general conversation, or something sensitive?"}


def route_from_intent(state: GraphState) -> str:
    intent = state.get("intent", "clarify")

    if intent == "product_details":
        return "product_node"
    if intent == "small_talk":
        return "small_talk_node"
    if intent == "sensitive_topic":
        return "sensitive_node"
    return "clarify_node"


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