import importlib
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory


@contextmanager
def _fake_span():
    """Stand-in for langfuse_client.start_as_current_span(...) as used
    (route calls span.update(...) inside the with-block)."""
    yield MagicMock()


@pytest.fixture
def chat_test_app(mocker, mock_es, mock_redis_binary):
    # Satisfy startup checks so the FastAPI app can import without needing a real
    # Elasticsearch index or Redis search backend.
    mock_es.indices.exists.return_value = True
    # seed_products_if_empty() (#39) checks es.count() at import time too —
    # report a non-empty index so it skips the (real, slow) seed path.
    mock_es.count.return_value = {"count": 1}
    mock_redis_binary.ft.return_value.info.return_value = {}
    # run_migrations() (Postgres Phase 1) also runs at import time and would
    # otherwise try to connect to a real database — no-op it here since this
    # test suite only cares about the chat route -> LangGraph wiring.
    mocker.patch("db.run_migrations")

    # Force a fresh import for each test so route-level patches apply against a
    # clean module instance instead of one cached by previous tests.
    for module_name in ["app.main", "app.routes.chat"]:
        sys.modules.pop(module_name, None)

    main = importlib.import_module("app.main")
    chat_module = importlib.import_module("app.routes.chat")

    # Rate limiting is not part of these tests; disable it so the assertions stay
    # focused on the chat route -> LangGraph wiring.
    main.app.state.limiter.enabled = False
    chat_module.limiter.enabled = False
    return main.app, chat_module


@pytest.mark.anyio
async def test_chat_route_uses_langgraph_and_preserves_response_shape(mocker, chat_test_app):
    app, chat_module = chat_test_app
    history = InMemoryChatMessageHistory()

    # Replace persistence and summarisation edges with deterministic fakes so the
    # test isolates the route contract rather than downstream integrations.
    mocker.patch.object(
        chat_module,
        "get_or_create_conversation",
        return_value={"history": history},
    )
    mocker.patch.object(chat_module, "maybe_summarise", return_value=(None, []))
    save_messages = mocker.patch.object(chat_module, "save_messages")

    # Freeze trace/callback creation so we can assert the exact config passed into
    # the graph invocation. create_trace_id is mocked to a non-hex placeholder,
    # so start_as_current_span/update_current_trace (which validate trace_id as
    # 32-char hex against the real Langfuse client) must be mocked too.
    mocker.patch.object(chat_module.langfuse_client, "create_trace_id", return_value="trace-123")
    mocker.patch.object(chat_module.langfuse_client, "start_as_current_span", side_effect=lambda **kw: _fake_span())
    mocker.patch.object(chat_module.langfuse_client, "update_current_trace")
    handler = object()
    mocker.patch.object(chat_module, "CallbackHandler", return_value=handler)

    # This is the core seam under test: the HTTP route should call LangGraph and
    # read the assistant text from the graph's "response" field.
    invoke = mocker.patch.object(
        chat_module.chat_graph,
        "invoke",
        return_value={"intent": "small_talk", "response": "GRAPH_WIRED_SENTINEL"},
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/chat",
            json={"session_id": "conv-1", "message": "hello"},
        )

    # The public API shape should remain unchanged even though the backend now
    # delegates response generation to LangGraph.
    assert response.status_code == 200
    assert response.json() == {
        "session_id": "conv-1",
        "response": "GRAPH_WIRED_SENTINEL",
        "message_count": 1,
        "trace_id": "trace-123",
    }

    # Prove the route sends the expected graph state and callback config, rather
    # than bypassing the graph or calling an older agent entry point.
    invoke.assert_called_once_with(
        {"input": "hello", "chat_history": []},
        config={"callbacks": [handler]},
    )

    # The route should still persist both sides of the exchange: the user message
    # and the assistant response produced by the graph.
    save_messages.assert_called_once()
    save_args = save_messages.call_args.args
    assert save_args[0] == "conv-1"
    assert len(save_args[1]) == 2
    assert save_args[1][0].content == "hello"
    assert save_args[1][1].content == "GRAPH_WIRED_SENTINEL"


@pytest.mark.anyio
async def test_chat_stream_route_uses_langgraph_and_preserves_sse_contract(mocker, chat_test_app):
    app, chat_module = chat_test_app
    history = InMemoryChatMessageHistory()

    # Keep the stream test isolated from real storage/summarisation so it only
    # verifies the route's streaming contract and graph integration.
    mocker.patch.object(
        chat_module,
        "get_or_create_conversation",
        return_value={"history": history},
    )
    mocker.patch.object(chat_module, "maybe_summarise", return_value=(None, []))
    save_messages = mocker.patch.object(chat_module, "save_messages")

    # Stabilise trace and callback dependencies so the streamed events and invoke
    # config are fully assertable. See the non-streaming test above for why
    # start_as_current_span/update_current_trace need mocking too.
    mocker.patch.object(chat_module.langfuse_client, "create_trace_id", return_value="trace-stream")
    mocker.patch.object(chat_module.langfuse_client, "start_as_current_span", side_effect=lambda **kw: _fake_span())
    mocker.patch.object(chat_module.langfuse_client, "update_current_trace")
    handler = object()
    mocker.patch.object(chat_module, "CallbackHandler", return_value=handler)

    # The streaming route currently performs one graph invocation and wraps that
    # result into the existing SSE protocol.
    invoke = mocker.patch.object(
        chat_module.chat_graph,
        "invoke",
        return_value={"intent": "product_details", "response": "GRAPH_STREAM_SENTINEL"},
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/chat/stream",
            json={"session_id": "conv-2", "message": "show me shoes"},
        )

    # Preserve the content type and SSE event order expected by the frontend:
    # trace metadata first, then assistant text, then the terminal [DONE] marker.
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    lines = [line for line in response.text.splitlines() if line]
    assert lines == [
        'data: {"trace_id": "trace-stream"}',
        'data: {"text": "GRAPH_STREAM_SENTINEL"}',
        "data: [DONE]",
    ]

    # The stream route should still route through LangGraph with the same message
    # payload and callback wiring as the non-streaming endpoint.
    invoke.assert_called_once_with(
        {"input": "show me shoes", "chat_history": []},
        config={"callbacks": [handler]},
    )

    # Even in streaming mode, the conversation history should persist the exact
    # user/assistant pair returned through the SSE response.
    save_messages.assert_called_once()
    save_args = save_messages.call_args.args
    assert save_args[0] == "conv-2"
    assert len(save_args[1]) == 2
    assert save_args[1][0].content == "show me shoes"
    assert save_args[1][1].content == "GRAPH_STREAM_SENTINEL"
