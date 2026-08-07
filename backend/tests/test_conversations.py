import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from conversations import load_preferences, maybe_summarise, save_preferences
from db import Base
from models_db import User


@pytest_asyncio.fixture
async def pg_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as s:
        yield s

    await engine.dispose()


def _messages(n):
    """n alternating Human/AI messages."""
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append(HumanMessage(content=f"user message {i}"))
        else:
            out.append(AIMessage(content=f"assistant message {i}"))
    return out


def _mock_llm(mocker, response_text="a summary"):
    llm = mocker.MagicMock()
    llm.ainvoke = mocker.AsyncMock(return_value=mocker.MagicMock(content=response_text))
    return llm


def _mock_structured_llm(mocker, summary="a summary", preferences="prefers budget electronics"):
    """A mock LLM whose .with_structured_output(...) returns a runnable
    producing a _SummaryAndPreferences-shaped result (#54). Also supports
    the plain .ainvoke(prompt) path (guest/no-user_id branch)."""
    llm = _mock_llm(mocker, response_text=summary)
    structured = mocker.MagicMock()
    structured.ainvoke = mocker.AsyncMock(
        return_value=mocker.MagicMock(summary=summary, preferences=preferences)
    )
    llm.with_structured_output = mocker.MagicMock(return_value=structured)
    return llm


async def test_under_threshold_returns_messages_unchanged(mocker, mock_conversations_redis):
    mock_conversations_redis.get.return_value = None
    mocker.patch("app.config.settings.conversation_summary_threshold", 20)
    llm = _mock_llm(mocker)

    messages = _messages(5)
    summary, recent = await maybe_summarise("conv1", messages, llm)

    assert summary is None
    assert recent == messages
    llm.ainvoke.assert_not_called()


async def test_under_threshold_still_returns_existing_summary(mocker, mock_conversations_redis):
    mock_conversations_redis.get.return_value = "an earlier summary"
    mocker.patch("app.config.settings.conversation_summary_threshold", 20)
    llm = _mock_llm(mocker)

    messages = _messages(5)
    summary, recent = await maybe_summarise("conv1", messages, llm)

    assert summary == "an earlier summary"
    assert recent == messages
    llm.ainvoke.assert_not_called()


async def test_at_threshold_triggers_summarisation_and_splits_messages(mocker, mock_conversations_redis):
    mock_conversations_redis.get.return_value = None
    mocker.patch("app.config.settings.conversation_summary_threshold", 20)
    llm = _mock_llm(mocker, "new summary")
    mock_set = mocker.patch("conversations.save_summary")
    mock_save_messages = mocker.patch("conversations.save_messages")

    messages = _messages(20)
    summary, recent = await maybe_summarise("conv1", messages, llm)

    assert summary == "new summary"
    # keep = threshold // 2 = 10
    assert recent == messages[-10:]
    mock_set.assert_called_once_with("conv1", "new summary")
    mock_save_messages.assert_called_once_with("conv1", messages[-10:])


async def test_prompt_includes_prior_summary_when_present(mocker, mock_conversations_redis):
    mock_conversations_redis.get.return_value = "earlier context about winter coats"
    mocker.patch("app.config.settings.conversation_summary_threshold", 20)
    llm = _mock_llm(mocker)
    mocker.patch("conversations.save_summary")
    mocker.patch("conversations.save_messages")

    await maybe_summarise("conv1", _messages(20), llm)

    prompt = llm.ainvoke.call_args.args[0]
    assert "Previous summary: earlier context about winter coats" in prompt


async def test_prompt_omits_prior_summary_line_when_absent(mocker, mock_conversations_redis):
    mock_conversations_redis.get.return_value = None
    mocker.patch("app.config.settings.conversation_summary_threshold", 20)
    llm = _mock_llm(mocker)
    mocker.patch("conversations.save_summary")
    mocker.patch("conversations.save_messages")

    await maybe_summarise("conv1", _messages(20), llm)

    prompt = llm.ainvoke.call_args.args[0]
    assert "Previous summary:" not in prompt


async def test_prompt_transcript_labels_human_and_ai_messages_correctly(mocker, mock_conversations_redis):
    mock_conversations_redis.get.return_value = None
    mocker.patch("app.config.settings.conversation_summary_threshold", 4)
    llm = _mock_llm(mocker)
    mocker.patch("conversations.save_summary")
    mocker.patch("conversations.save_messages")

    # 4 messages, threshold 4 -> keep = 2, summarise the first 2 (index 0,1)
    await maybe_summarise("conv1", _messages(4), llm)

    prompt = llm.ainvoke.call_args.args[0]
    assert "User: user message 0" in prompt
    assert "Assistant: assistant message 1" in prompt


# --- Long-term memory / preferences (#54) ---


async def test_load_preferences_returns_none_when_no_row_exists(pg_session):
    user = User(email="a@example.com", name="A")
    pg_session.add(user)
    await pg_session.commit()

    assert await load_preferences(pg_session, user.id) is None


async def test_save_then_load_preferences_round_trips(pg_session):
    user = User(email="a@example.com", name="A")
    pg_session.add(user)
    await pg_session.commit()

    await save_preferences(pg_session, user.id, "prefers budget electronics")

    assert await load_preferences(pg_session, user.id) == "prefers budget electronics"


async def test_save_preferences_overwrites_existing_row(pg_session):
    user = User(email="a@example.com", name="A")
    pg_session.add(user)
    await pg_session.commit()

    await save_preferences(pg_session, user.id, "prefers budget electronics")
    await save_preferences(pg_session, user.id, "prefers premium electronics")

    assert await load_preferences(pg_session, user.id) == "prefers premium electronics"


async def test_maybe_summarise_without_user_id_never_touches_preferences(mocker, mock_conversations_redis):
    """Guest path (user_id=None) must behave exactly as before #54 — no
    structured output, no preferences read/write."""
    mock_conversations_redis.get.return_value = None
    mocker.patch("app.config.settings.conversation_summary_threshold", 20)
    llm = _mock_structured_llm(mocker)
    mocker.patch("conversations.save_summary")
    mocker.patch("conversations.save_messages")
    mock_save_preferences = mocker.patch("conversations.save_preferences")

    summary, _ = await maybe_summarise("conv1", _messages(20), llm)

    llm.with_structured_output.assert_not_called()
    mock_save_preferences.assert_not_called()
    assert summary == "a summary"  # the plain llm.ainvoke(prompt).content path


async def test_maybe_summarise_with_user_id_updates_preferences(mocker, mock_conversations_redis, pg_session):
    user = User(email="a@example.com", name="A")
    pg_session.add(user)
    await pg_session.commit()

    mock_conversations_redis.get.return_value = None
    mocker.patch("app.config.settings.conversation_summary_threshold", 20)
    llm = _mock_structured_llm(mocker, summary="new summary", preferences="prefers budget electronics")
    mocker.patch("conversations.save_summary")
    mocker.patch("conversations.save_messages")

    summary, recent = await maybe_summarise(
        "conv1", _messages(20), llm, pg_session=pg_session, user_id=user.id
    )

    assert summary == "new summary"
    assert recent == _messages(20)[-10:]
    assert await load_preferences(pg_session, user.id) == "prefers budget electronics"


async def test_maybe_summarise_under_threshold_does_not_touch_preferences_even_with_user_id(
    mocker, mock_conversations_redis, pg_session
):
    user = User(email="a@example.com", name="A")
    pg_session.add(user)
    await pg_session.commit()

    mock_conversations_redis.get.return_value = None
    mocker.patch("app.config.settings.conversation_summary_threshold", 20)
    llm = _mock_structured_llm(mocker)

    await maybe_summarise("conv1", _messages(5), llm, pg_session=pg_session, user_id=user.id)

    llm.with_structured_output.assert_not_called()
    assert await load_preferences(pg_session, user.id) is None


async def test_maybe_summarise_prompt_includes_existing_preferences(mocker, mock_conversations_redis, pg_session):
    user = User(email="a@example.com", name="A")
    pg_session.add(user)
    await pg_session.commit()
    await save_preferences(pg_session, user.id, "prefers budget electronics")

    mock_conversations_redis.get.return_value = None
    mocker.patch("app.config.settings.conversation_summary_threshold", 20)
    llm = _mock_structured_llm(mocker)
    mocker.patch("conversations.save_summary")
    mocker.patch("conversations.save_messages")

    await maybe_summarise("conv1", _messages(20), llm, pg_session=pg_session, user_id=user.id)

    structured_runnable = llm.with_structured_output.return_value
    prompt = structured_runnable.ainvoke.call_args.args[0]
    assert "Known preferences so far: prefers budget electronics" in prompt
