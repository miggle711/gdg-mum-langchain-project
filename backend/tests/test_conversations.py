from langchain_core.messages import AIMessage, HumanMessage

from conversations import maybe_summarise


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
    llm.invoke.return_value = mocker.MagicMock(content=response_text)
    return llm


def test_under_threshold_returns_messages_unchanged(mocker, mock_conversations_redis):
    mock_conversations_redis.get.return_value = None
    mocker.patch("app.config.settings.conversation_summary_threshold", 20)
    llm = _mock_llm(mocker)

    messages = _messages(5)
    summary, recent = maybe_summarise("conv1", messages, llm)

    assert summary is None
    assert recent == messages
    llm.invoke.assert_not_called()


def test_under_threshold_still_returns_existing_summary(mocker, mock_conversations_redis):
    mock_conversations_redis.get.return_value = "an earlier summary"
    mocker.patch("app.config.settings.conversation_summary_threshold", 20)
    llm = _mock_llm(mocker)

    messages = _messages(5)
    summary, recent = maybe_summarise("conv1", messages, llm)

    assert summary == "an earlier summary"
    assert recent == messages
    llm.invoke.assert_not_called()


def test_at_threshold_triggers_summarisation_and_splits_messages(mocker, mock_conversations_redis):
    mock_conversations_redis.get.return_value = None
    mocker.patch("app.config.settings.conversation_summary_threshold", 20)
    llm = _mock_llm(mocker, "new summary")
    mock_set = mocker.patch("conversations.save_summary")
    mock_save_messages = mocker.patch("conversations.save_messages")

    messages = _messages(20)
    summary, recent = maybe_summarise("conv1", messages, llm)

    assert summary == "new summary"
    # keep = threshold // 2 = 10
    assert recent == messages[-10:]
    mock_set.assert_called_once_with("conv1", "new summary")
    mock_save_messages.assert_called_once_with("conv1", messages[-10:])


def test_prompt_includes_prior_summary_when_present(mocker, mock_conversations_redis):
    mock_conversations_redis.get.return_value = "earlier context about winter coats"
    mocker.patch("app.config.settings.conversation_summary_threshold", 20)
    llm = _mock_llm(mocker)
    mocker.patch("conversations.save_summary")
    mocker.patch("conversations.save_messages")

    maybe_summarise("conv1", _messages(20), llm)

    prompt = llm.invoke.call_args.args[0]
    assert "Previous summary: earlier context about winter coats" in prompt


def test_prompt_omits_prior_summary_line_when_absent(mocker, mock_conversations_redis):
    mock_conversations_redis.get.return_value = None
    mocker.patch("app.config.settings.conversation_summary_threshold", 20)
    llm = _mock_llm(mocker)
    mocker.patch("conversations.save_summary")
    mocker.patch("conversations.save_messages")

    maybe_summarise("conv1", _messages(20), llm)

    prompt = llm.invoke.call_args.args[0]
    assert "Previous summary:" not in prompt


def test_prompt_transcript_labels_human_and_ai_messages_correctly(mocker, mock_conversations_redis):
    mock_conversations_redis.get.return_value = None
    mocker.patch("app.config.settings.conversation_summary_threshold", 4)
    llm = _mock_llm(mocker)
    mocker.patch("conversations.save_summary")
    mocker.patch("conversations.save_messages")

    # 4 messages, threshold 4 -> keep = 2, summarise the first 2 (index 0,1)
    maybe_summarise("conv1", _messages(4), llm)

    prompt = llm.invoke.call_args.args[0]
    assert "User: user message 0" in prompt
    assert "Assistant: assistant message 1" in prompt
