import os
import sys

os.environ.setdefault("GOOGLE_API_KEY", "test-key")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture
def mock_es(mocker):
    """A mocked Elasticsearch client, patched into search.get_es()."""
    es = mocker.MagicMock()
    mocker.patch("search.get_es", return_value=es)
    return es


@pytest.fixture
def mock_reranker(mocker):
    """A mocked cross-encoder, patched into search.get_reranker()."""
    reranker = mocker.MagicMock()
    mocker.patch("search.get_reranker", return_value=reranker)
    return reranker


@pytest.fixture
def mock_redis_binary(mocker):
    """A mocked binary Redis client, patched into cache._get_redis_binary()."""
    r = mocker.MagicMock()
    mocker.patch("cache._get_redis_binary", return_value=r)
    return r


@pytest.fixture
def mock_redis(mocker):
    """A mocked string Redis client, patched into cache._get_redis()."""
    r = mocker.MagicMock()
    mocker.patch("cache._get_redis", return_value=r)
    return r


@pytest.fixture
def mock_conversations_redis(mocker):
    """A mocked Redis client for conversations.py, which imports
    _get_redis directly (`from cache import _get_redis`) and so has its
    own module-level binding that patching cache._get_redis won't reach.
    """
    r = mocker.MagicMock()
    mocker.patch("conversations._get_redis", return_value=r)
    return r
