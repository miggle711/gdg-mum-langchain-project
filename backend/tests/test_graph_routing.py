# backend/tests/test_graph.py
import os

import pytest
from langchain_core.messages import HumanMessage


LIVE_LLM_INTENT_CASES = [
    ("Do you have any running shoes under $80?", "product_details"),
    ("Can you compare the best laptops for college?", "product_details"),
    ("Hey there, how's your day going?", "small_talk"),
    ("Thanks, that's all I needed today.", "small_talk"),
    ("I want to hurt someone who scammed me.", "sensitive_topic"),
    ("asdf qwerty", "clarify"),
]

# Tests routing
def test_classify_intent_routes_known_prompts(mocker):
    import app.graph as graph

    mock_classifier = mocker.Mock()
    mock_classifier.invoke.side_effect = [
        graph.IntentClassification(intent="product_details"),
        graph.IntentClassification(intent="small_talk"),
        graph.IntentClassification(intent="sensitive_topic"),
        graph.IntentClassification(intent="product_details"),
        graph.IntentClassification(intent="clarify"),
    ]
    mocker.patch.object(graph, "_intent_classifier", mock_classifier)

    assert graph.classify_intent({"input": "Do you have any running shoes under $80?"})["intent"] == "product_details"
    assert graph.classify_intent({"input": "Hey there, how's your day going?"})["intent"] == "small_talk"
    assert graph.classify_intent({"input": "I want to hurt someone who scammed me."})["intent"] == "sensitive_topic"
    assert graph.classify_intent({"input": "Can you compare the best laptops for college?"})["intent"] == "product_details"

    history_state = {
        "input": "What about one in blue?",
        "chat_history": [HumanMessage(content="Show me waterproof jackets for hiking.")],
    }
    assert graph.classify_intent(history_state)["intent"] == "clarify"

    assert mock_classifier.invoke.call_args_list[4].args[0] == {
        "input": "What about one in blue?",
        "chat_history": [HumanMessage(content="Show me waterproof jackets for hiking.")],
    }

# Tests fallback behavior
def test_classify_intent_falls_back_to_clarify(mocker):
    import app.graph as graph

    class InvalidClassification:
        intent = "not_a_real_intent"

    mock_classifier = mocker.Mock()
    mock_classifier.invoke.return_value = InvalidClassification()
    mocker.patch.object(graph, "_intent_classifier", mock_classifier)

    assert graph.classify_intent({"input": "asdf qwerty"})["intent"] == "clarify"

# Tests fallback behavior when LLM errors
def test_classify_intent_falls_back_to_clarify_when_llm_errors(mocker):
    import app.graph as graph

    mock_classifier = mocker.Mock()
    mock_classifier.invoke.side_effect = RuntimeError("temporary model failure")
    mocker.patch.object(graph, "_intent_classifier", mock_classifier)

    assert graph.classify_intent({"input": "I'm looking for a gift but not sure what kind"})["intent"] == "clarify"


def test_graph_compiles():
    import app.graph as graph

    assert graph.chat_graph is not None


def test_product_node_uses_product_agent(mocker):
    import app.graph as graph

    mock_invoke = mocker.patch.object(
        graph,
        "_invoke_product_agent",
        return_value={"output": "Here are three laptop options under $900."},
    )

    state = {
        "input": "Show me laptops under $900",
        "chat_history": [HumanMessage(content="I need something for school.")],
    }

    assert graph.product_node(state) == {
        "response": "Here are three laptop options under $900."
    }
    mock_invoke.assert_called_once_with(state, config=None)


def test_product_node_falls_back_when_agent_returns_no_output(mocker):
    import app.graph as graph

    mocker.patch.object(graph, "_invoke_product_agent", return_value={})

    assert graph.product_node({"input": "Find a coffee grinder"}) == {
        "response": "I apologize, but I'm having trouble generating a response at the moment."
    }

# Tests live LLM classification (requires a real GOOGLE_API_KEY)
@pytest.mark.llm
@pytest.mark.skipif(
    os.getenv("GOOGLE_API_KEY") in (None, "", "test-key"),
    reason="A real GOOGLE_API_KEY is required.",
)
@pytest.mark.parametrize(("prompt", "expected_intent"), LIVE_LLM_INTENT_CASES)
def test_classify_intent_with_live_llm(prompt, expected_intent):
    import app.graph as graph

    if os.getenv("GOOGLE_API_KEY") in (None, "", "test-key"):
        pytest.skip("A real GOOGLE_API_KEY is required for live LLM classification tests.")

    assert graph.classify_intent({"input": prompt})["intent"] == expected_intent

# Tests live LLM classification with chat history (requires a real GOOGLE_API_KEY)
@pytest.mark.llm
@pytest.mark.skipif(
    os.getenv("GOOGLE_API_KEY") in (None, "", "test-key"),
    reason="A real GOOGLE_API_KEY is required.",
)
def test_classify_intent_with_live_llm_uses_chat_history():
    import app.graph as graph

    if os.getenv("GOOGLE_API_KEY") in (None, "", "test-key"):
        pytest.skip("A real GOOGLE_API_KEY is required for live LLM classification tests.")

    state = {
        "input": "What about one in blue?",
        "chat_history": [HumanMessage(content="Show me waterproof jackets for hiking.")],
    }

    assert graph.classify_intent(state)["intent"] == "product_details"
