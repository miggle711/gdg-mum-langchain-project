from search import semantic_search


def _es_response(hits):
    return {"hits": {"hits": hits}}


def _hit(score=1.0, **source):
    return {"_score": score, "_source": source}


def test_cache_hit_skips_es_and_reranker_and_truncates_to_limit(mocker, mock_es, mock_reranker):
    cached_results = [{"id": f"p{i}"} for i in range(10)]
    mocker.patch("cache.get_cached_search", return_value=cached_results)
    set_cached = mocker.patch("cache.set_cached_search")

    results = semantic_search("cozy blanket", [0.1] * 768, limit=3)

    assert results == cached_results[:3]
    mock_es.search.assert_not_called()
    mock_reranker.predict.assert_not_called()
    set_cached.assert_not_called()


def test_cache_miss_runs_hybrid_query_with_expected_boosts(mocker, mock_es, mock_reranker):
    mocker.patch("cache.get_cached_search", return_value=None)
    mocker.patch("cache.set_cached_search")
    mock_es.search.return_value = _es_response([])
    mock_reranker.predict.return_value = mocker.MagicMock(tolist=lambda: [])

    semantic_search("cozy blanket", [0.1] * 768, limit=5)

    body = mock_es.search.call_args.kwargs["body"]
    should = body["query"]["bool"]["should"]
    multi_match = next(c["multi_match"] for c in should if "multi_match" in c)
    knn = next(c["knn"] for c in should if "knn" in c)

    assert multi_match["boost"] == 0.5
    assert knn["boost"] == 4.0
    assert knn["field"] == "embedding"
    assert body["size"] == max(20, 5 * 4)


def test_no_candidates_returns_empty_without_reranking(mocker, mock_es, mock_reranker):
    mocker.patch("cache.get_cached_search", return_value=None)
    set_cached = mocker.patch("cache.set_cached_search")
    mock_es.search.return_value = _es_response([])

    results = semantic_search("cozy blanket", [0.1] * 768, limit=5)

    assert results == []
    mock_reranker.predict.assert_not_called()
    set_cached.assert_not_called()


def test_candidates_are_reranked_sorted_and_capped_to_limit(mocker, mock_es, mock_reranker):
    mocker.patch("cache.get_cached_search", return_value=None)
    set_cached = mocker.patch("cache.set_cached_search")
    mock_es.search.return_value = _es_response([
        _hit(id="low", name="Low match", description="desc", price=10, rating=4.0, reviews=5, category="Books"),
        _hit(id="high", name="High match", description="desc", price=20, rating=4.0, reviews=5, category="Books"),
    ])
    # reranker scores "low" candidate lower than "high", despite ES order being low-first
    mock_reranker.predict.return_value = mocker.MagicMock(tolist=lambda: [0.2, 0.9])

    results = semantic_search("query", [0.1] * 768, limit=1)

    assert len(results) == 1
    assert results[0]["id"] == "high"
    assert results[0]["similarity"] == 0.9
    assert "description" not in results[0]
    set_cached.assert_called_once()


def test_reranker_receives_query_paired_with_name_and_description(mocker, mock_es, mock_reranker):
    mocker.patch("cache.get_cached_search", return_value=None)
    mocker.patch("cache.set_cached_search")
    mock_es.search.return_value = _es_response([
        _hit(id="p1", name="Wireless Headphones", description="Noise cancelling", price=99, rating=4.5, reviews=10, category="Electronics"),
    ])
    mock_reranker.predict.return_value = mocker.MagicMock(tolist=lambda: [0.5])

    semantic_search("headphones", [0.1] * 768, limit=5)

    pairs = mock_reranker.predict.call_args.args[0]
    assert pairs == [("headphones", "Wireless Headphones. Noise cancelling")]
