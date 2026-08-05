import pytest


@pytest.fixture
def graph_module(mock_embedding_model):
    """graph.py imports tools.py, which lazily loads the BGE embedding model
    via search.get_embedding_model() — mocked here so importing graph.py (or
    actually calling resolve_product_reference) doesn't try to download/load
    the real model.
    """
    import app.graph as graph
    return graph


async def test_resolve_product_reference_returns_top_result_id(mocker, graph_module, mock_embedding_model):
    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mocker.patch("tools.semantic_search", return_value=[
        {
            "id": "p1", "name": "Blue Waterproof Jacket", "price": 79.99, "originalprice": None,
            "rating": 4.5, "reviews": 12, "category_name": "Outdoor", "similarity": 0.91,
        },
    ])

    result = await graph_module.resolve_product_reference("the blue jacket")

    assert result == "p1"


async def test_resolve_product_reference_requests_only_the_top_match(mocker, graph_module, mock_embedding_model):
    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mock_search = mocker.patch("tools.semantic_search", return_value=[])

    await graph_module.resolve_product_reference("the blue jacket")

    assert mock_search.call_args.kwargs["limit"] == 1


async def test_resolve_product_reference_returns_none_when_no_results(mocker, graph_module, mock_embedding_model):
    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mocker.patch("tools.semantic_search", return_value=[])

    result = await graph_module.resolve_product_reference("a nonexistent gadget")

    assert result is None


async def test_resolve_product_reference_returns_none_on_search_error(mocker, graph_module, mock_embedding_model):
    # semantic_search_impl catches its own exceptions and returns
    # {"error": ..., "results": []} — resolve_product_reference relies on
    # that instead of duplicating error handling.
    mock_embedding_model.encode.side_effect = RuntimeError("model error")

    result = await graph_module.resolve_product_reference("anything")

    assert result is None
