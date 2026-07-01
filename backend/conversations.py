import json
import logging
from typing import List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, messages_from_dict, messages_to_dict
from app.config import settings
from cache import _get_redis

logger = logging.getLogger(__name__)

CONVERSATION_TTL_SECONDS = settings.conversation_ttl_seconds


def save_messages(conversation_id: str, messages: List[BaseMessage]) -> None:
    r = _get_redis()
    key = f"conversation:{conversation_id}"
    r.set(key, json.dumps(messages_to_dict(messages)), ex=CONVERSATION_TTL_SECONDS)
    logger.info("Saved %d messages for conversation '%s'", len(messages), conversation_id)


def load_messages(conversation_id: str) -> List[BaseMessage]:
    r = _get_redis()
    key = f"conversation:{conversation_id}"
    data = r.get(key)
    if data is None:
        logger.info("No session found in Redis for conversation '%s', starting fresh.", conversation_id)
        return []
    messages = messages_from_dict(json.loads(data))
    logger.info("Loaded %d messages for conversation '%s' from Redis.", len(messages), conversation_id)
    return messages


def save_summary(conversation_id: str, summary: str) -> None:
    r = _get_redis()
    r.set(f"conversation:{conversation_id}:summary", summary, ex=CONVERSATION_TTL_SECONDS)


def load_summary(conversation_id: str) -> Optional[str]:
    r = _get_redis()
    return r.get(f"conversation:{conversation_id}:summary")


def maybe_summarise(conversation_id: str, messages: List[BaseMessage], llm) -> tuple[Optional[str], List[BaseMessage]]:
    threshold = settings.conversation_summary_threshold
    existing_summary = load_summary(conversation_id)

    if len(messages) < threshold:
        return existing_summary, messages

    keep = threshold // 2
    to_summarise = messages[:-keep]
    recent = messages[-keep:]

    logger.info(
        "Summarising %d messages for conversation '%s' (keeping last %d verbatim)...",
        len(to_summarise), conversation_id, keep,
    )

    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in to_summarise
    )

    prior_context = f"Previous summary: {existing_summary}\n\n" if existing_summary else ""
    prompt = (
        f"{prior_context}"
        f"Summarise the following conversation in 3-5 sentences. "
        f"Focus on what the user was looking for, any products discussed, and any preferences expressed. "
        f"Write from the assistant's perspective.\n\n{history_text}"
    )

    summary = llm.invoke(prompt).content
    logger.info("Summary generated for conversation '%s': %s", conversation_id, summary[:80])

    save_summary(conversation_id, summary)
    save_messages(conversation_id, recent)

    return summary, recent
