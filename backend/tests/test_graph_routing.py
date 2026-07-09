# backend/tests/test_graph.py
def test_classify_intent_routes_known_prompts(mocker):
    import app.graph as graph

    assert graph.classify_intent({"input": "search product prices"})["intent"] == "product_details"
    assert graph.classify_intent({"input": "this is small talk"})["intent"] == "small_talk"
    assert graph.classify_intent({"input": "this is a sensitive topic"})["intent"] == "sensitive_topic"


def test_classify_intent_falls_back_to_clarify():
    import app.graph as graph

    assert graph.classify_intent({"input": "asdf qwerty"})["intent"] == "clarify"


def test_graph_compiles():
    import app.graph as graph

    assert graph.chat_graph is not None