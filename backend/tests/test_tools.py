import json

import pytest


@pytest.fixture
def tools_module(mock_embedding_model):
    """tools.py gets its embedding model via search.get_embedding_model()
    (a lazy singleton), mocked by the mock_embedding_model fixture so
    importing tools.py doesn't try to download/load the real BGE model.
    """
    import tools
    return tools


def test_query_products_impl_only_includes_provided_filters(mocker, tools_module):
    mock_query = mocker.patch("tools.query_products", return_value=[])

    tools_module.query_products_impl(category="Electronics", price_max=50)

    filters = mock_query.call_args.args[0]
    assert filters == {"category": "Electronics", "price_max": 50}


def test_query_products_impl_formats_price_and_rating_as_strings(mocker, tools_module):
    mocker.patch("tools.query_products", return_value=[{
        "id": "p1",
        "name": "Wireless Headphones",
        "price": 99.5,
        "originalprice": 129.995,
        "rating": 4.5,
        "reviews": 10,
        "category_name": "Electronics",
    }])

    result = json.loads(tools_module.query_products_impl(category="Electronics"))

    product = result["results"][0]
    assert product["price"] == "$99.50"
    assert product["original_price"] == "$130.00"
    assert product["rating"] == "4.5/5"
    assert result["count"] == 1


def test_query_products_impl_handles_missing_original_price(mocker, tools_module):
    mocker.patch("tools.query_products", return_value=[{
        "id": "p1", "name": "Cable", "price": 9.99, "originalprice": None,
        "rating": 3.0, "reviews": 2, "category_name": "Electronics",
    }])

    result = json.loads(tools_module.query_products_impl(search="cable"))

    assert result["results"][0]["original_price"] is None


def test_query_products_impl_empty_results_returns_message(mocker, tools_module):
    mocker.patch("tools.query_products", return_value=[])

    result = json.loads(tools_module.query_products_impl(search="nonexistent"))

    assert result["results"] == []
    assert "message" in result


def test_query_products_impl_catches_exceptions(mocker, tools_module):
    mocker.patch("tools.query_products", side_effect=RuntimeError("ES is down"))

    result = json.loads(tools_module.query_products_impl(search="anything"))

    assert result["results"] == []
    assert result["error"] == "ES is down"


def test_semantic_search_impl_uses_bge_retrieval_prefix(mocker, tools_module, mock_embedding_model):
    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mock_search = mocker.patch("tools.semantic_search", return_value=[])

    tools_module.semantic_search_impl("cozy winter blanket")

    encode_call = mock_embedding_model.encode.call_args
    assert encode_call.args[0] == "Represent this sentence for searching relevant passages: cozy winter blanket"
    assert encode_call.kwargs["normalize_embeddings"] is True
    # original (unprefixed) query text is passed through to semantic_search
    assert mock_search.call_args.args[0] == "cozy winter blanket"


def test_semantic_search_impl_defaults_limit_to_5_when_none(mocker, tools_module, mock_embedding_model):
    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mock_search = mocker.patch("tools.semantic_search", return_value=[])

    tools_module.semantic_search_impl("blanket", limit=None)

    assert mock_search.call_args.kwargs["limit"] == 5


def test_semantic_search_impl_formats_similarity_as_percent(mocker, tools_module, mock_embedding_model):
    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)
    mocker.patch("tools.semantic_search", return_value=[{
        "id": "p1", "name": "Blanket", "price": 29.99, "originalprice": None,
        "rating": 4.2, "reviews": 8, "category_name": "Home", "similarity": 0.873,
    }])

    result = json.loads(tools_module.semantic_search_impl("cozy blanket"))

    assert result["results"][0]["similarity"] == "87%"


def test_semantic_search_impl_catches_exceptions(mocker, tools_module, mock_embedding_model):
    mock_embedding_model.encode.side_effect = RuntimeError("model error")

    result = json.loads(tools_module.semantic_search_impl("blanket"))

    assert result["results"] == []
    assert result["error"] == "model error"


def test_list_categories_impl_returns_name_and_icon_only(mocker, tools_module):
    mocker.patch("tools.get_categories", return_value=[
        {"name": "Electronics", "icon": "📦", "extra_field": "ignored"},
    ])

    result = json.loads(tools_module.list_categories_impl())

    assert result["categories"] == [{"name": "Electronics", "icon": "📦"}]


def test_list_categories_impl_catches_exceptions(mocker, tools_module):
    mocker.patch("tools.get_categories", side_effect=RuntimeError("ES is down"))

    result = json.loads(tools_module.list_categories_impl())

    assert result["categories"] == []
    assert result["error"] == "ES is down"


def test_product_tools_have_expected_names(tools_module):
    names = {t.name for t in tools_module.PRODUCT_TOOLS}
    assert names == {"semantic_search", "query_products", "list_categories"}
