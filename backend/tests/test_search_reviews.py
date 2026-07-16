from search import semantic_search_reviews


def _es_response(hits):
    return {"hits": {"hits": hits}}


def _hit(score=1.0, **source):
    return {"_score": score, "_source": source}


def test_cache_hit_skips_es_and_reranker_and_truncates_to_limit(mocker, mock_es, mock_reranker):
    cached_results = [{"product_id": f"p{i}"} for i in range(10)]
    mocker.patch("cache.get_cached_review_search", return_value=cached_results)
    set_cached = mocker.patch("cache.set_cached_review_search")

    results = semantic_search_reviews("battery life", [0.1] * 768, limit=3)

    assert results == cached_results[:3]
    mock_es.search.assert_not_called()
    mock_reranker.predict.assert_not_called()
    set_cached.assert_not_called()


def test_cache_miss_runs_hybrid_query_with_expected_boosts(mocker, mock_es, mock_reranker):
    mocker.patch("cache.get_cached_review_search", return_value=None)
    mocker.patch("cache.set_cached_review_search")
    mock_es.search.return_value = _es_response([])
    mock_reranker.predict.return_value = mocker.MagicMock(tolist=lambda: [])

    semantic_search_reviews("battery life", [0.1] * 768, limit=5)

    call_kwargs = mock_es.search.call_args.kwargs
    assert call_kwargs["index"] == "reviews"
    body = call_kwargs["body"]
    should = body["query"]["bool"]["should"]
    multi_match = next(c["multi_match"] for c in should if "multi_match" in c)
    knn = next(c["knn"] for c in should if "knn" in c)

    assert multi_match["boost"] == 0.5
    assert multi_match["fields"] == ["title^2", "text"]
    assert knn["boost"] == 4.0
    assert knn["field"] == "embedding"
    assert body["size"] == max(20, 5 * 4)


def test_no_candidates_returns_empty_without_reranking(mocker, mock_es, mock_reranker):
    mocker.patch("cache.get_cached_review_search", return_value=None)
    set_cached = mocker.patch("cache.set_cached_review_search")
    mock_es.search.return_value = _es_response([])

    results = semantic_search_reviews("battery life", [0.1] * 768, limit=5)

    assert results == []
    mock_reranker.predict.assert_not_called()
    set_cached.assert_not_called()


def test_candidates_are_reranked_sorted_and_capped_to_limit(mocker, mock_es, mock_reranker):
    mocker.patch("cache.get_cached_review_search", return_value=None)
    set_cached = mocker.patch("cache.set_cached_review_search")
    mock_es.search.return_value = _es_response([
        _hit(product_id="p1", product_name="Widget", title="Low match", text="meh", rating=3.0, verified_purchase=True, helpful_vote=1),
        _hit(product_id="p2", product_name="Gadget", title="High match", text="great", rating=5.0, verified_purchase=True, helpful_vote=9),
    ])
    # reranker scores the first candidate lower than the second, despite ES order
    mock_reranker.predict.return_value = mocker.MagicMock(tolist=lambda: [0.2, 0.9])

    results = semantic_search_reviews("query", [0.1] * 768, limit=1)

    assert len(results) == 1
    assert results[0]["product_id"] == "p2"
    assert results[0]["product_name"] == "Gadget"
    assert results[0]["similarity"] == 0.9
    set_cached.assert_called_once()


def test_reranker_receives_query_paired_with_title_and_text(mocker, mock_es, mock_reranker):
    mocker.patch("cache.get_cached_review_search", return_value=None)
    mocker.patch("cache.set_cached_review_search")
    mock_es.search.return_value = _es_response([
        _hit(product_id="p1", title="Great battery", text="Lasts all day", rating=5.0, verified_purchase=True, helpful_vote=3),
    ])
    mock_reranker.predict.return_value = mocker.MagicMock(tolist=lambda: [0.5])

    semantic_search_reviews("battery life", [0.1] * 768, limit=5)

    pairs = mock_reranker.predict.call_args.args[0]
    assert pairs == [("battery life", "Great battery. Lasts all day")]
