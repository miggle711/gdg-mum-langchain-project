import os
import sys

os.environ.setdefault("GOOGLE_API_KEY", "test-key")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture
def mock_es(mocker):
    """A mocked AsyncElasticsearch client, patched into search.get_es().
    Use mocker.AsyncMock(), not MagicMock() — search.py awaits es.search(),
    es.indices.exists(), etc., so their return values must be awaitable.
    """
    es = mocker.AsyncMock()
    mocker.patch("search.get_es", return_value=es)
    return es


@pytest.fixture
def mock_reranker(mocker):
    """A mocked cross-encoder, patched into search.get_reranker()."""
    reranker = mocker.MagicMock()
    mocker.patch("search.get_reranker", return_value=reranker)
    return reranker


@pytest.fixture
def mock_embedding_model(mocker):
    """A mocked SentenceTransformer, patched into search.get_embedding_model()."""
    model = mocker.MagicMock()
    mocker.patch("search.get_embedding_model", return_value=model)
    return model


@pytest.fixture
def mock_redis_binary(mocker):
    """A mocked binary Redis client, patched into cache._get_redis_binary().
    r.ft(index) is a sync factory call (like r.pipeline()) — only the
    methods on its return value (.info() / .search() / .create_index())
    are actually awaited by cache.py, so those are AsyncMock while r.ft
    itself and r.pipeline() stay sync-returning MagicMocks. Using a plain
    AsyncMock for r would auto-spec r.ft itself as async too, breaking the
    chained r.ft(...).info() call shape.
    """
    r = mocker.MagicMock()
    ft_client = mocker.MagicMock()
    ft_client.info = mocker.AsyncMock()
    ft_client.search = mocker.AsyncMock()
    ft_client.create_index = mocker.AsyncMock()
    r.ft = mocker.MagicMock(return_value=ft_client)

    pipe = mocker.MagicMock()
    pipe.execute = mocker.AsyncMock()
    r.pipeline = mocker.MagicMock(return_value=pipe)

    mocker.patch("cache._get_redis_binary", return_value=r)
    return r


@pytest.fixture
def mock_redis(mocker):
    """A mocked string Redis client, patched into cache._get_redis()."""
    r = mocker.AsyncMock()
    mocker.patch("cache._get_redis", return_value=r)
    return r


@pytest.fixture
def mock_conversations_redis(mocker):
    """A mocked Redis client for conversations.py, which imports
    _get_redis directly (`from cache import _get_redis`) and so has its
    own module-level binding that patching cache._get_redis won't reach.
    """
    r = mocker.AsyncMock()
    mocker.patch("conversations._get_redis", return_value=r)
    return r


@pytest.fixture
def mock_db_session(mocker):
    """A mocked SQLAlchemy AsyncSession, patched into db.get_session().
    Use mocker.AsyncMock(), not MagicMock() — db.py's session is async, so
    methods like session.commit()/session.execute() get awaited by callers.
    A MagicMock's return value isn't awaitable, so `await mock.commit()`
    raises TypeError: object MagicMock can't be used in 'await' expression.
    """
    session = mocker.AsyncMock()
    mocker.patch("db.get_session", return_value=session)
    return session
