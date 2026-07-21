from search import query_products


def _es_response(hits):
    return {"hits": {"hits": hits}}


def _hit(**source):
    return {"_source": source}


async def test_no_filters_uses_match_all(mock_es):
    mock_es.search.return_value = _es_response([])

    await query_products({})

    body = mock_es.search.call_args.kwargs["body"]
    assert body["query"] == {"match_all": {}}


async def test_search_filter_builds_multi_match_must_clause(mock_es):
    mock_es.search.return_value = _es_response([])

    await query_products({"search": "headphones"})

    body = mock_es.search.call_args.kwargs["body"]
    assert body["query"]["bool"]["must"] == [
        {"multi_match": {"query": "headphones", "fields": ["name^2", "description"]}}
    ]
    assert "filter" not in body["query"]["bool"]


async def test_category_filter_builds_term_filter_clause(mock_es):
    mock_es.search.return_value = _es_response([])

    await query_products({"category": "Electronics"})

    body = mock_es.search.call_args.kwargs["body"]
    assert {"term": {"category": "Electronics"}} in body["query"]["bool"]["filter"]
    assert "must" not in body["query"]["bool"]


async def test_price_and_rating_filters_build_range_clauses(mock_es):
    mock_es.search.return_value = _es_response([])

    await query_products({"price_min": 10, "price_max": 100, "rating_min": 4.0})

    filters = mock_es.search.call_args.kwargs["body"]["query"]["bool"]["filter"]
    assert {"range": {"price": {"gte": 10}}} in filters
    assert {"range": {"price": {"lte": 100}}} in filters
    assert {"range": {"rating": {"gte": 4.0}}} in filters


async def test_combined_filters_produce_must_and_filter_clauses(mock_es):
    mock_es.search.return_value = _es_response([])

    await query_products({"search": "headphones", "category": "Electronics", "price_max": 50})

    body = mock_es.search.call_args.kwargs["body"]
    assert len(body["query"]["bool"]["must"]) == 1
    assert len(body["query"]["bool"]["filter"]) == 2


async def test_always_sorts_by_rating_then_reviews_desc(mock_es):
    mock_es.search.return_value = _es_response([])

    await query_products({"search": "headphones"})

    body = mock_es.search.call_args.kwargs["body"]
    assert body["sort"] == [{"rating": "desc"}, {"reviews": "desc"}]
    assert body["size"] == 20


async def test_result_shaping_renames_fields_and_handles_missing_optional(mock_es):
    mock_es.search.return_value = _es_response([
        _hit(
            id="p1",
            name="Wireless Headphones",
            price=99.99,
            original_price=129.99,
            rating=4.5,
            reviews=1200,
            category="Electronics",
        ),
        _hit(
            id="p2",
            name="USB Cable",
            price=9.99,
            rating=3.8,
            reviews=50,
            category="Electronics",
        ),
    ])

    results = await query_products({"category": "Electronics"})

    assert results[0] == {
        "id": "p1",
        "name": "Wireless Headphones",
        "price": 99.99,
        "originalprice": 129.99,
        "rating": 4.5,
        "reviews": 1200,
        "category_name": "Electronics",
    }
    # original_price absent in source -> originalprice is None, not a KeyError
    assert results[1]["originalprice"] is None
