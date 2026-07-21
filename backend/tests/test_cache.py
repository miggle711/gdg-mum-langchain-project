import struct

import pytest

import cache
from cache import _embedding_to_bytes, get_cached_search, set_cached_search


def _search_result(docs):
    result = type("Result", (), {})()
    result.docs = docs
    return result


def _doc(score, results_json, doc_id="search_cache:1"):
    doc = type("Doc", (), {})()
    doc.score = score
    doc.results = results_json
    doc.id = doc_id
    return doc


def test_embedding_to_bytes_round_trips_via_struct_unpack():
    embedding = [0.1, -0.5, 3.25]

    packed = _embedding_to_bytes(embedding)
    unpacked = list(struct.unpack(f"{len(embedding)}f", packed))

    # FLOAT32 packing loses some precision vs Python's native float64
    assert unpacked == pytest.approx(embedding, abs=1e-6)


async def test_no_cached_docs_returns_none(mock_redis_binary):
    mock_redis_binary.ft.return_value.search.return_value = _search_result([])

    assert await get_cached_search([0.1] * 768) is None


async def test_distance_within_threshold_is_a_cache_hit(mock_redis_binary):
    # default threshold = 1.0 - 0.92 = 0.08
    mock_redis_binary.ft.return_value.search.return_value = _search_result(
        [_doc(score=0.05, results_json='[{"id": "p1"}]')]
    )

    result = await get_cached_search([0.1] * 768)

    assert result == [{"id": "p1"}]


async def test_distance_just_past_threshold_is_a_cache_miss(mock_redis_binary):
    # threshold = 1.0 - 0.92 = 0.08 (approx, subject to float rounding)
    mock_redis_binary.ft.return_value.search.return_value = _search_result(
        [_doc(score=0.09, results_json='[{"id": "p1"}]')]
    )

    assert await get_cached_search([0.1] * 768) is None


async def test_set_cached_search_stores_embedding_and_results_with_ttl(mocker, mock_redis_binary):
    pipe = mock_redis_binary.pipeline.return_value

    await set_cached_search("cozy blanket", [0.1] * 768, [{"id": "p1"}])

    hset_call = pipe.hset.call_args
    key = hset_call.args[0]
    mapping = hset_call.kwargs["mapping"]

    assert key.startswith(cache.CACHE_PREFIX)
    assert mapping["embedding"] == _embedding_to_bytes([0.1] * 768)
    assert mapping["results"] == '[{"id": "p1"}]'
    pipe.expire.assert_called_once_with(key, cache.SEARCH_CACHE_TTL_SECONDS)
    pipe.execute.assert_called_once()
