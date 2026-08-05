import json

from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage


# --- retrieve_data (RD) ---
# No more search-vs-details branch: always searches, and additionally
# fetches a full "detail" record when product_reference resolves to a real
# product. See issue57.md's "redesign: merge product_search/product_details"
# for why the old upfront branch was replaced.

async def test_retrieve_data_always_runs_search_with_raw_input(mocker, mock_embedding_model):
    import app.graph as graph

    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mock_search = mocker.patch("tools.semantic_search", return_value=[
        {
            "id": "p1", "name": "Running Shoes", "price": 59.99, "originalprice": None,
            "rating": 4.2, "reviews": 30, "category_name": "Footwear", "similarity": 0.88,
        },
    ])

    result = await graph.retrieve_data({"intent": "product_query", "input": "running shoes under $80"})

    assert mock_search.call_args.args[0] == "running shoes under $80"
    retrieved = json.loads(result["retrieved_data"])
    assert retrieved == {
        "results": [
            {
                "id": "p1", "name": "Running Shoes", "price": "$59.99", "original_price": None,
                "rating": "4.2/5", "reviews": 30, "category": "Footwear", "similarity": "88%",
            },
        ],
        "count": 1,
        "detail": None,
    }


async def test_retrieve_data_passes_extracted_filters_through(mocker, mock_embedding_model):
    import app.graph as graph

    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mock_search = mocker.patch("tools.semantic_search", return_value=[])

    state = {
        "intent": "product_query", "input": "electronics under $20",
        "category": "Electronics", "price_max": 20,
    }
    await graph.retrieve_data(state)

    assert mock_search.call_args.kwargs["filters"] == {"category": "Electronics", "price_max": 20}


async def test_retrieve_data_no_filters_extracted_passes_none(mocker, mock_embedding_model):
    import app.graph as graph

    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mock_search = mocker.patch("tools.semantic_search", return_value=[])

    await graph.retrieve_data({"intent": "product_query", "input": "something cozy for winter"})

    assert mock_search.call_args.kwargs["filters"] is None


async def test_retrieve_data_empty_results_and_no_reference(mocker, mock_embedding_model):
    import app.graph as graph

    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mocker.patch("tools.semantic_search", return_value=[])

    result = await graph.retrieve_data({"intent": "product_query", "input": "something nonexistent"})

    assert json.loads(result["retrieved_data"]) == {"results": [], "count": 0, "detail": None}


async def test_retrieve_data_with_reference_also_fetches_detail(mocker, mock_embedding_model):
    import app.graph as graph

    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mocker.patch("tools.semantic_search", return_value=[])
    mocker.patch.object(graph, "resolve_product_reference", AsyncMock(return_value="p1"))
    mock_get_product = mocker.patch(
        "cart_tools.get_product_impl",
        AsyncMock(return_value=json.dumps({"id": "p1", "name": "Blue Jacket", "price": 79.99})),
    )

    state = {"intent": "product_query", "input": "tell me more about the blue jacket", "product_reference": "the blue jacket"}
    result = await graph.retrieve_data(state)

    graph.resolve_product_reference.assert_called_once_with("the blue jacket")
    mock_get_product.assert_called_once_with("p1")
    retrieved = json.loads(result["retrieved_data"])
    assert retrieved["detail"] == {"id": "p1", "name": "Blue Jacket", "price": 79.99}


async def test_retrieve_data_without_a_reference_skips_detail_lookup(mocker, mock_embedding_model):
    import app.graph as graph

    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mocker.patch("tools.semantic_search", return_value=[])
    resolve_mock = mocker.patch.object(graph, "resolve_product_reference", AsyncMock())
    get_product_mock = mocker.patch("cart_tools.get_product_impl", AsyncMock())

    result = await graph.retrieve_data({"intent": "product_query", "input": "show me jackets"})

    resolve_mock.assert_not_called()
    get_product_mock.assert_not_called()
    assert json.loads(result["retrieved_data"])["detail"] is None


async def test_retrieve_data_reference_that_does_not_resolve_omits_detail(mocker, mock_embedding_model):
    import app.graph as graph

    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mocker.patch("tools.semantic_search", return_value=[])
    mocker.patch.object(graph, "resolve_product_reference", AsyncMock(return_value=None))
    get_product_mock = mocker.patch("cart_tools.get_product_impl", AsyncMock())

    result = await graph.retrieve_data({
        "intent": "product_query", "input": "tell me about the made-up item", "product_reference": "a made-up item",
    })

    get_product_mock.assert_not_called()
    assert json.loads(result["retrieved_data"])["detail"] is None


async def test_retrieve_data_reference_resolves_but_product_not_found(mocker, mock_embedding_model):
    import app.graph as graph

    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mocker.patch("tools.semantic_search", return_value=[])
    mocker.patch.object(graph, "resolve_product_reference", AsyncMock(return_value="ghost-id"))
    mocker.patch(
        "cart_tools.get_product_impl",
        AsyncMock(return_value=json.dumps({"error": "Product not found"})),
    )

    result = await graph.retrieve_data({
        "intent": "product_query", "input": "tell me about the deleted item", "product_reference": "a deleted item",
    })

    assert json.loads(result["retrieved_data"])["detail"] is None


# --- validate_results (VR) / route_from_validation ---

def test_validate_results_node_is_a_pass_through():
    import app.graph as graph

    retrieved = json.dumps({"results": [{"id": "p1"}], "count": 1, "detail": None})
    assert graph.validate_results({"retrieved_data": retrieved}) == {}


def test_route_from_validation_routes_to_grounded_response_when_results_present():
    import app.graph as graph

    retrieved = json.dumps({"results": [{"id": "p1"}], "count": 1, "detail": None})
    assert graph.route_from_validation({"retrieved_data": retrieved}) == "generate_grounded_response"


def test_route_from_validation_routes_to_grounded_response_when_only_detail_present():
    import app.graph as graph

    # A resolved reference with an empty broad-search list still counts as
    # "found something" — the point of the redesign is that detail alone is
    # enough, no need for the search list to also be non-empty.
    retrieved = json.dumps({"results": [], "count": 0, "detail": {"id": "p1", "name": "Blue Jacket"}})
    assert graph.route_from_validation({"retrieved_data": retrieved}) == "generate_grounded_response"


def test_route_from_validation_routes_to_clarify_when_nothing_found():
    import app.graph as graph

    retrieved = json.dumps({"results": [], "count": 0, "detail": None})
    assert graph.route_from_validation({"retrieved_data": retrieved}) == "clarify_node"


def test_route_from_validation_treats_missing_retrieved_data_as_no_results():
    import app.graph as graph

    assert graph.route_from_validation({}) == "clarify_node"


# --- generate_grounded_response (GR) ---

async def test_generate_grounded_response_returns_llm_content(mocker):
    import app.graph as graph

    mocker.patch.object(
        graph, "_grounded_response_chain",
        mocker.Mock(ainvoke=AsyncMock(return_value=AIMessage(content="We have three great jackets in stock."))),
    )

    result = await graph.generate_grounded_response({
        "input": "show me jackets",
        "retrieved_data": json.dumps({"results": [{"id": "p1", "name": "Jacket"}], "count": 1, "detail": None}),
    })

    assert result == {"response": "We have three great jackets in stock."}


async def test_generate_grounded_response_passes_input_history_and_retrieved_data(mocker):
    import app.graph as graph

    mock_chain = mocker.Mock(ainvoke=AsyncMock(return_value=AIMessage(content="Here you go.")))
    mocker.patch.object(graph, "_grounded_response_chain", mock_chain)

    retrieved = json.dumps({"results": [{"id": "p1"}], "count": 1, "detail": None})
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

async def test_chat_graph_end_to_end_product_query_browsing(mocker, mock_embedding_model):
    import app.graph as graph

    mocker.patch.object(graph, "_intent_entity_extractor", mocker.Mock(
        ainvoke=AsyncMock(return_value=graph.IntentEntityExtraction(intent="product_query")),
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

    assert result["intent"] == "product_query"
    assert result["response"] == "We have great running shoes for $59.99!"


async def test_chat_graph_end_to_end_product_query_no_results_routes_to_clarify(mocker, mock_embedding_model):
    import app.graph as graph

    mocker.patch.object(graph, "_intent_entity_extractor", mocker.Mock(
        ainvoke=AsyncMock(return_value=graph.IntentEntityExtraction(intent="product_query")),
    ))
    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mocker.patch("tools.semantic_search", return_value=[])

    result = await graph.chat_graph.ainvoke({"input": "something nonexistent", "chat_history": [], "session_id": "s1"})

    assert result["response"] == "response from graph - clarify node placeholder"


async def test_chat_graph_end_to_end_product_query_with_resolved_reference(mocker, mock_embedding_model):
    import app.graph as graph

    mocker.patch.object(graph, "_intent_entity_extractor", mocker.Mock(
        ainvoke=AsyncMock(return_value=graph.IntentEntityExtraction(
            intent="product_query", product_reference="the blue jacket",
        )),
    ))
    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mocker.patch("tools.semantic_search", return_value=[])
    mocker.patch.object(graph, "resolve_product_reference", AsyncMock(return_value="p1"))
    mocker.patch(
        "cart_tools.get_product_impl",
        AsyncMock(return_value=json.dumps({"id": "p1", "name": "Blue Jacket", "price": 79.99})),
    )
    mocker.patch.object(graph, "_grounded_response_chain", mocker.Mock(
        ainvoke=AsyncMock(return_value=AIMessage(content="The Blue Jacket is $79.99.")),
    ))

    result = await graph.chat_graph.ainvoke({"input": "tell me about the blue jacket", "chat_history": [], "session_id": "s1"})

    assert result["intent"] == "product_query"
    assert result["response"] == "The Blue Jacket is $79.99."
