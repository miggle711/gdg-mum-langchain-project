import json

from unittest.mock import AsyncMock


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
