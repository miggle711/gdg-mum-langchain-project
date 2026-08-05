# backend/tests/test_graph.py
import os
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage

# Placeholder-string matching (e.g. checking for "test-key") is brittle —
# CI uses a different placeholder ("dummy-key-for-tests") than local dev's
# conftest.py default ("test-key"), and neither actually looks like a real
# key. Real Google API keys start with "AIza"; anything else is a stand-in.
# The primary guard is still `pytest -m "not llm"` in CI — this is a
# secondary safety net for anyone running the full suite locally without
# that flag but without a real key either.
_HAS_REAL_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").startswith("AIza")

LIVE_LLM_INTENT_CASES = [
    ("Do you have any running shoes under $80?", "product_search"),
    ("Can you compare the best laptops for college?", "product_search"),
    ("Hey there, how's your day going?", "fallback"),
    ("Thanks, that's all I needed today.", "fallback"),
    ("I want to hurt someone who scammed me.", "unsafe"),
    ("asdf qwerty", "clarify"),
]

# Tests routing
async def test_extract_intent_and_entities_routes_known_prompts(mocker):
    import app.graph as graph

    mock_extractor = mocker.Mock()
    mock_extractor.ainvoke = AsyncMock(side_effect=[
        graph.IntentEntityExtraction(intent="product_search"),
        graph.IntentEntityExtraction(intent="fallback"),
        graph.IntentEntityExtraction(intent="unsafe"),
        graph.IntentEntityExtraction(intent="product_search"),
        graph.IntentEntityExtraction(intent="clarify"),
    ])
    mocker.patch.object(graph, "_intent_entity_extractor", mock_extractor)

    assert (await graph.extract_intent_and_entities({"input": "Do you have any running shoes under $80?"}))["intent"] == "product_search"
    assert (await graph.extract_intent_and_entities({"input": "Hey there, how's your day going?"}))["intent"] == "fallback"
    assert (await graph.extract_intent_and_entities({"input": "I want to hurt someone who scammed me."}))["intent"] == "unsafe"
    assert (await graph.extract_intent_and_entities({"input": "Can you compare the best laptops for college?"}))["intent"] == "product_search"

    history_state = {
        "input": "What about one in blue?",
        "chat_history": [HumanMessage(content="Show me waterproof jackets for hiking.")],
    }
    assert (await graph.extract_intent_and_entities(history_state))["intent"] == "clarify"

    assert mock_extractor.ainvoke.call_args_list[4].args[0] == {
        "input": "What about one in blue?",
        "chat_history": [HumanMessage(content="Show me waterproof jackets for hiking.")],
    }


async def test_extract_intent_and_entities_returns_cart_action_entities(mocker):
    import app.graph as graph

    mock_extractor = mocker.Mock()
    mock_extractor.ainvoke = AsyncMock(return_value=graph.IntentEntityExtraction(
        intent="cart_action",
        product_reference="the blue jacket",
        quantity=2,
        cart_action_type="add",
    ))
    mocker.patch.object(graph, "_intent_entity_extractor", mock_extractor)

    result = await graph.extract_intent_and_entities({"input": "Add 2 of the blue jacket to my cart"})

    assert result == {
        "intent": "cart_action",
        "product_reference": "the blue jacket",
        "quantity": 2,
        "cart_action_type": "add",
    }


# Tests fallback behavior
async def test_extract_intent_and_entities_falls_back_to_clarify(mocker):
    import app.graph as graph

    class InvalidExtraction:
        intent = "not_a_real_intent"
        product_reference = None
        quantity = None
        cart_action_type = None

    mock_extractor = mocker.Mock()
    mock_extractor.ainvoke = AsyncMock(return_value=InvalidExtraction())
    mocker.patch.object(graph, "_intent_entity_extractor", mock_extractor)

    result = await graph.extract_intent_and_entities({"input": "asdf qwerty"})

    assert result["intent"] == "clarify"
    # Extraction ran successfully (just returned an unrecognized label) —
    # not an infra failure, so no error flag (contrast with the LLM-errors
    # test below).
    assert "error" not in result

# Tests fallback behavior when LLM errors
async def test_extract_intent_and_entities_falls_back_to_clarify_when_llm_errors(mocker):
    import app.graph as graph

    mock_extractor = mocker.Mock()
    mock_extractor.ainvoke = AsyncMock(side_effect=RuntimeError("temporary model failure"))
    mocker.patch.object(graph, "_intent_entity_extractor", mock_extractor)

    result = await graph.extract_intent_and_entities({"input": "I'm looking for a gift but not sure what kind"})

    assert result["intent"] == "clarify"
    # A real classifier failure (e.g. quota exhaustion) must be distinguishable
    # from a genuine ambiguous-message clarify, not silently look like a normal
    # successful reply to the caller (#77).
    assert result["error"] is True


def test_graph_compiles():
    import app.graph as graph

    assert graph.chat_graph is not None


def test_route_from_intent_routes_each_intent_to_the_expected_node():
    import app.graph as graph

    assert graph.route_from_intent({"intent": "product_search"}) == "product_search_node"
    assert graph.route_from_intent({"intent": "product_details"}) == "product_details_node"
    # TEMPORARY (#57): cart_action has no real destination yet — clarify_node
    # is a placeholder until the cart-action nodes are built next.
    assert graph.route_from_intent({"intent": "cart_action"}) == "clarify_node"
    assert graph.route_from_intent({"intent": "unsafe"}) == "sensitive_node"
    assert graph.route_from_intent({"intent": "fallback"}) == "small_talk_node"
    assert graph.route_from_intent({"intent": "clarify"}) == "clarify_node"
    assert graph.route_from_intent({}) == "clarify_node"

# Tests live LLM classification (requires a real GOOGLE_API_KEY)
@pytest.mark.llm
@pytest.mark.skipif(
    not _HAS_REAL_GOOGLE_API_KEY,
    reason="A real GOOGLE_API_KEY is required.",
)
@pytest.mark.parametrize(("prompt", "expected_intent"), LIVE_LLM_INTENT_CASES)
async def test_extract_intent_and_entities_with_live_llm(prompt, expected_intent):
    import app.graph as graph

    assert (await graph.extract_intent_and_entities({"input": prompt}))["intent"] == expected_intent

# Tests live LLM classification with chat history (requires a real GOOGLE_API_KEY)
@pytest.mark.llm
@pytest.mark.skipif(
    not _HAS_REAL_GOOGLE_API_KEY,
    reason="A real GOOGLE_API_KEY is required.",
)
async def test_extract_intent_and_entities_with_live_llm_uses_chat_history():
    import app.graph as graph

    state = {
        "input": "What about one in blue?",
        "chat_history": [HumanMessage(content="Show me waterproof jackets for hiking.")],
    }

    assert (await graph.extract_intent_and_entities(state))["intent"] == "product_search"
