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
    ("Do you have any running shoes under $80?", "product_details"),
    ("Can you compare the best laptops for college?", "product_details"),
    ("Hey there, how's your day going?", "small_talk"),
    ("Thanks, that's all I needed today.", "small_talk"),
    ("I want to hurt someone who scammed me.", "sensitive_topic"),
    ("asdf qwerty", "clarify"),
]

# Tests routing
async def test_classify_intent_routes_known_prompts(mocker):
    import app.graph as graph

    mock_classifier = mocker.Mock()
    mock_classifier.ainvoke = AsyncMock(side_effect=[
        graph.IntentClassification(intent="product_details"),
        graph.IntentClassification(intent="small_talk"),
        graph.IntentClassification(intent="sensitive_topic"),
        graph.IntentClassification(intent="product_details"),
        graph.IntentClassification(intent="clarify"),
    ])
    mocker.patch.object(graph, "_intent_classifier", mock_classifier)

    assert (await graph.classify_intent({"input": "Do you have any running shoes under $80?"}))["intent"] == "product_details"
    assert (await graph.classify_intent({"input": "Hey there, how's your day going?"}))["intent"] == "small_talk"
    assert (await graph.classify_intent({"input": "I want to hurt someone who scammed me."}))["intent"] == "sensitive_topic"
    assert (await graph.classify_intent({"input": "Can you compare the best laptops for college?"}))["intent"] == "product_details"

    history_state = {
        "input": "What about one in blue?",
        "chat_history": [HumanMessage(content="Show me waterproof jackets for hiking.")],
    }
    assert (await graph.classify_intent(history_state))["intent"] == "clarify"

    assert mock_classifier.ainvoke.call_args_list[4].args[0] == {
        "input": "What about one in blue?",
        "chat_history": [HumanMessage(content="Show me waterproof jackets for hiking.")],
    }

# Tests fallback behavior
async def test_classify_intent_falls_back_to_clarify(mocker):
    import app.graph as graph

    class InvalidClassification:
        intent = "not_a_real_intent"

    mock_classifier = mocker.Mock()
    mock_classifier.ainvoke = AsyncMock(return_value=InvalidClassification())
    mocker.patch.object(graph, "_intent_classifier", mock_classifier)

    result = await graph.classify_intent({"input": "asdf qwerty"})

    assert result["intent"] == "clarify"
    # Classifier ran successfully (just returned an unrecognized label) —
    # not an infra failure, so no error flag (contrast with the LLM-errors
    # test below).
    assert "error" not in result

# Tests fallback behavior when LLM errors
async def test_classify_intent_falls_back_to_clarify_when_llm_errors(mocker):
    import app.graph as graph

    mock_classifier = mocker.Mock()
    mock_classifier.ainvoke = AsyncMock(side_effect=RuntimeError("temporary model failure"))
    mocker.patch.object(graph, "_intent_classifier", mock_classifier)

    result = await graph.classify_intent({"input": "I'm looking for a gift but not sure what kind"})

    assert result["intent"] == "clarify"
    # A real classifier failure (e.g. quota exhaustion) must be distinguishable
    # from a genuine ambiguous-message clarify, not silently look like a normal
    # successful reply to the caller (#77).
    assert result["error"] is True


def test_graph_compiles():
    import app.graph as graph

    assert graph.chat_graph is not None


async def test_product_node_uses_product_agent(mocker):
    import app.graph as graph

    mock_invoke = mocker.patch.object(
        graph,
        "_invoke_product_agent",
        AsyncMock(return_value={"output": "Here are three laptop options under $900."}),
    )

    state = {
        "input": "Show me laptops under $900",
        "chat_history": [HumanMessage(content="I need something for school.")],
    }

    assert (await graph.product_node(state)) == {
        "response": "Here are three laptop options under $900."
    }
    mock_invoke.assert_called_once_with(state, config=None)


async def test_product_node_falls_back_when_agent_returns_no_output(mocker):
    import app.graph as graph

    mocker.patch.object(graph, "_invoke_product_agent", AsyncMock(return_value={}))

    assert (await graph.product_node({"input": "Find a coffee grinder"})) == {
        "response": "I apologize, but I'm having trouble generating a response at the moment."
    }

# Tests live LLM classification (requires a real GOOGLE_API_KEY)
@pytest.mark.llm
@pytest.mark.skipif(
    not _HAS_REAL_GOOGLE_API_KEY,
    reason="A real GOOGLE_API_KEY is required.",
)
@pytest.mark.parametrize(("prompt", "expected_intent"), LIVE_LLM_INTENT_CASES)
async def test_classify_intent_with_live_llm(prompt, expected_intent):
    import app.graph as graph

    assert (await graph.classify_intent({"input": prompt}))["intent"] == expected_intent

# Tests live LLM classification with chat history (requires a real GOOGLE_API_KEY)
@pytest.mark.llm
@pytest.mark.skipif(
    not _HAS_REAL_GOOGLE_API_KEY,
    reason="A real GOOGLE_API_KEY is required.",
)
async def test_classify_intent_with_live_llm_uses_chat_history():
    import app.graph as graph

    state = {
        "input": "What about one in blue?",
        "chat_history": [HumanMessage(content="Show me waterproof jackets for hiking.")],
    }

    assert (await graph.classify_intent(state))["intent"] == "product_details"
