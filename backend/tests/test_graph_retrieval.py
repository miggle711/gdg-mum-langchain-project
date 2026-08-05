import json

from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage


async def test_product_search_node_and_product_details_node_are_pass_throughs():
    import app.graph as graph

    assert (await graph.product_search_node({"input": "shoes"})) == {}
    assert (await graph.product_details_node({"input": "tell me more"})) == {}


async def test_retrieve_data_product_search_uses_semantic_search_with_raw_input(mocker, mock_embedding_model):
    import app.graph as graph

    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mock_search = mocker.patch("tools.semantic_search", return_value=[
        {
            "id": "p1", "name": "Running Shoes", "price": 59.99, "originalprice": None,
            "rating": 4.2, "reviews": 30, "category_name": "Footwear", "similarity": 0.88,
        },
    ])

    result = await graph.retrieve_data({"intent": "product_search", "input": "running shoes under $80"})

    assert mock_search.call_args.args[0] == "running shoes under $80"
    retrieved = json.loads(result["retrieved_data"])
    assert retrieved["count"] == 1
    assert retrieved["results"][0]["id"] == "p1"


async def test_retrieve_data_product_search_empty_results(mocker, mock_embedding_model):
    import app.graph as graph

    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mocker.patch("tools.semantic_search", return_value=[])

    result = await graph.retrieve_data({"intent": "product_search", "input": "something nonexistent"})

    assert json.loads(result["retrieved_data"]) == {"results": [], "count": 0}


async def test_retrieve_data_product_details_resolves_reference_then_looks_up_product(mocker):
    import app.graph as graph

    mocker.patch.object(graph, "resolve_product_reference", AsyncMock(return_value="p1"))
    mock_get_product = mocker.patch(
        "cart_tools.get_product_impl",
        AsyncMock(return_value=json.dumps({"id": "p1", "name": "Blue Jacket", "price": 79.99})),
    )

    state = {"intent": "product_details", "product_reference": "the blue jacket"}
    result = await graph.retrieve_data(state)

    graph.resolve_product_reference.assert_called_once_with("the blue jacket")
    mock_get_product.assert_called_once_with("p1")
    retrieved = json.loads(result["retrieved_data"])
    assert retrieved == {"count": 1, "results": [{"id": "p1", "name": "Blue Jacket", "price": 79.99}]}


async def test_retrieve_data_product_details_without_a_reference_skips_lookup(mocker):
    import app.graph as graph

    resolve_mock = mocker.patch.object(graph, "resolve_product_reference", AsyncMock())
    get_product_mock = mocker.patch("cart_tools.get_product_impl", AsyncMock())

    result = await graph.retrieve_data({"intent": "product_details"})

    resolve_mock.assert_not_called()
    get_product_mock.assert_not_called()
    assert json.loads(result["retrieved_data"]) == {"results": [], "count": 0}


async def test_retrieve_data_product_details_reference_that_does_not_resolve(mocker):
    import app.graph as graph

    mocker.patch.object(graph, "resolve_product_reference", AsyncMock(return_value=None))
    get_product_mock = mocker.patch("cart_tools.get_product_impl", AsyncMock())

    result = await graph.retrieve_data({"intent": "product_details", "product_reference": "a made-up item"})

    get_product_mock.assert_not_called()
    assert json.loads(result["retrieved_data"]) == {"results": [], "count": 0}


async def test_retrieve_data_product_details_product_not_found(mocker):
    import app.graph as graph

    mocker.patch.object(graph, "resolve_product_reference", AsyncMock(return_value="ghost-id"))
    mocker.patch(
        "cart_tools.get_product_impl",
        AsyncMock(return_value=json.dumps({"error": "Product not found"})),
    )

    result = await graph.retrieve_data({"intent": "product_details", "product_reference": "a deleted item"})

    assert json.loads(result["retrieved_data"]) == {"results": [], "count": 0}


def test_validate_results_node_is_a_pass_through():
    import app.graph as graph

    retrieved = json.dumps({"results": [{"id": "p1"}], "count": 1})
    assert graph.validate_results({"retrieved_data": retrieved}) == {}


def test_route_from_validation_routes_to_grounded_response_when_results_present():
    import app.graph as graph

    retrieved = json.dumps({"results": [{"id": "p1"}], "count": 1})
    assert graph.route_from_validation({"retrieved_data": retrieved}) == "generate_grounded_response"


def test_route_from_validation_routes_to_clarify_when_no_results():
    import app.graph as graph

    retrieved = json.dumps({"results": [], "count": 0})
    assert graph.route_from_validation({"retrieved_data": retrieved}) == "clarify_node"


def test_route_from_validation_treats_missing_retrieved_data_as_no_results():
    import app.graph as graph

    assert graph.route_from_validation({}) == "clarify_node"


async def test_generate_grounded_response_returns_llm_content(mocker):
    import app.graph as graph

    mocker.patch.object(
        graph, "_grounded_response_chain",
        mocker.Mock(ainvoke=AsyncMock(return_value=AIMessage(content="We have three great jackets in stock."))),
    )

    result = await graph.generate_grounded_response({
        "input": "show me jackets",
        "retrieved_data": json.dumps({"results": [{"id": "p1", "name": "Jacket"}], "count": 1}),
    })

    assert result == {"response": "We have three great jackets in stock."}


async def test_generate_grounded_response_passes_input_history_and_retrieved_data(mocker):
    import app.graph as graph

    mock_chain = mocker.Mock(ainvoke=AsyncMock(return_value=AIMessage(content="Here you go.")))
    mocker.patch.object(graph, "_grounded_response_chain", mock_chain)

    retrieved = json.dumps({"results": [{"id": "p1"}], "count": 1})
    history = [HumanMessage(content="Looking for a jacket")]
    await graph.generate_grounded_response({"input": "show me jackets", "chat_history": history, "retrieved_data": retrieved})

    assert mock_chain.ainvoke.call_args.args[0] == {
        "input": "show me jackets",
        "chat_history": history,
        "retrieved_data": retrieved,
    }


async def test_generate_grounded_response_falls_back_when_llm_errors(mocker):
    import app.graph as graph

    mocker.patch.object(
        graph, "_grounded_response_chain",
        mocker.Mock(ainvoke=AsyncMock(side_effect=RuntimeError("model unavailable"))),
    )

    result = await graph.generate_grounded_response({"input": "show me jackets", "retrieved_data": "{}"})

    assert result == {"response": "I apologize, but I'm having trouble generating a response at the moment."}


async def test_generate_grounded_response_falls_back_when_content_empty(mocker):
    import app.graph as graph

    mocker.patch.object(
        graph, "_grounded_response_chain",
        mocker.Mock(ainvoke=AsyncMock(return_value=AIMessage(content=""))),
    )

    result = await graph.generate_grounded_response({"input": "show me jackets", "retrieved_data": "{}"})

    assert result == {"response": "I apologize, but I'm having trouble generating a response at the moment."}


# --- End-to-end: proves the actual wiring in build_chat_graph() is correct,
# not just each node in isolation (a typo'd node-name string in an edge
# wouldn't be caught by any of the unit tests above). ---

async def test_chat_graph_end_to_end_product_search_with_results(mocker, mock_embedding_model):
    import app.graph as graph

    mocker.patch.object(graph, "_intent_entity_extractor", mocker.Mock(
        ainvoke=AsyncMock(return_value=graph.IntentEntityExtraction(intent="product_search")),
    ))
    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mocker.patch("tools.semantic_search", return_value=[
        {
            "id": "p1", "name": "Running Shoes", "price": 59.99, "originalprice": None,
            "rating": 4.2, "reviews": 30, "category_name": "Footwear", "similarity": 0.88,
        },
    ])
    mocker.patch.object(graph, "_grounded_response_chain", mocker.Mock(
        ainvoke=AsyncMock(return_value=AIMessage(content="We have great running shoes for $59.99!")),
    ))

    result = await graph.chat_graph.ainvoke({"input": "running shoes", "chat_history": [], "session_id": "s1"})

    assert result["intent"] == "product_search"
    assert result["response"] == "We have great running shoes for $59.99!"


async def test_chat_graph_end_to_end_product_search_no_results_routes_to_clarify(mocker, mock_embedding_model):
    import app.graph as graph

    mocker.patch.object(graph, "_intent_entity_extractor", mocker.Mock(
        ainvoke=AsyncMock(return_value=graph.IntentEntityExtraction(intent="product_search")),
    ))
    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mocker.patch("tools.semantic_search", return_value=[])

    result = await graph.chat_graph.ainvoke({"input": "something nonexistent", "chat_history": [], "session_id": "s1"})

    assert result["response"] == "response from graph - clarify node placeholder"


async def test_chat_graph_end_to_end_product_details(mocker, mock_embedding_model):
    import app.graph as graph

    mocker.patch.object(graph, "_intent_entity_extractor", mocker.Mock(
        ainvoke=AsyncMock(return_value=graph.IntentEntityExtraction(
            intent="product_details", product_reference="the blue jacket",
        )),
    ))
    mocker.patch.object(graph, "resolve_product_reference", AsyncMock(return_value="p1"))
    mocker.patch(
        "cart_tools.get_product_impl",
        AsyncMock(return_value=json.dumps({"id": "p1", "name": "Blue Jacket", "price": 79.99})),
    )
    mocker.patch.object(graph, "_grounded_response_chain", mocker.Mock(
        ainvoke=AsyncMock(return_value=AIMessage(content="The Blue Jacket is $79.99.")),
    ))

    result = await graph.chat_graph.ainvoke({"input": "tell me about the blue jacket", "chat_history": [], "session_id": "s1"})

    assert result["intent"] == "product_details"
    assert result["response"] == "The Blue Jacket is $79.99."
