import json
import logging
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, messages_from_dict, messages_to_dict
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from cache import _get_redis
from models_db import UserPreferences

logger = logging.getLogger(__name__)

CONVERSATION_TTL_SECONDS = settings.conversation_ttl_seconds


async def save_messages(session_id: str, messages: List[BaseMessage]) -> None:
    r = _get_redis()
    key = f"conversation:{session_id}"
    await r.set(key, json.dumps(messages_to_dict(messages)), ex=CONVERSATION_TTL_SECONDS)
    logger.info("Saved %d messages for conversation '%s'", len(messages), session_id)


async def load_messages(session_id: str) -> List[BaseMessage]:
    r = _get_redis()
    key = f"conversation:{session_id}"
    data = await r.get(key)
    if data is None:
        logger.info("No session found in Redis for conversation '%s', starting fresh.", session_id)
        return []
    messages = messages_from_dict(json.loads(data))
    logger.info("Loaded %d messages for conversation '%s' from Redis.", len(messages), session_id)
    return messages


async def save_summary(session_id: str, summary: str) -> None:
    r = _get_redis()
    await r.set(f"conversation:{session_id}:summary", summary, ex=CONVERSATION_TTL_SECONDS)


async def load_summary(session_id: str) -> Optional[str]:
    r = _get_redis()
    return await r.get(f"conversation:{session_id}:summary")


async def load_preferences(session: AsyncSession, user_id: int) -> Optional[str]:
    """Reads the freeform long-term preferences blob for a real (authenticated)
    user (#54). Returns None if the user has no preferences recorded yet."""
    row = await session.get(UserPreferences, user_id)
    return row.preferences if row else None


async def save_preferences(session: AsyncSession, user_id: int, preferences: str) -> None:
    row = await session.get(UserPreferences, user_id)
    if row is None:
        session.add(UserPreferences(user_id=user_id, preferences=preferences))
    else:
        row.preferences = preferences
    await session.commit()


class _SummaryAndPreferences(BaseModel):
    summary: str = Field(description="3-5 sentence summary of the conversation, from the assistant's perspective")
    preferences: str = Field(
        description="Updated freeform notes on this customer's durable preferences (e.g. brands, "
        "price sensitivity, categories of interest) merged with any prior preferences given. "
        "Keep it concise — a few sentences, not a list of every message."
    )


async def maybe_summarise(
    session_id: str,
    messages: List[BaseMessage],
    llm,
    *,
    pg_session: AsyncSession | None = None,
    user_id: int | None = None,
) -> tuple[Optional[str], List[BaseMessage]]:
    threshold = settings.conversation_summary_threshold
    existing_summary = await load_summary(session_id)

    if len(messages) < threshold:
        return existing_summary, messages

    keep = threshold // 2
    to_summarise = messages[:-keep]
    recent = messages[-keep:]

    logger.info(
        "Summarising %d messages for conversation '%s' (keeping last %d verbatim)...",
        len(to_summarise), session_id, keep,
    )

    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in to_summarise
    )

    prior_context = f"Previous summary: {existing_summary}\n\n" if existing_summary else ""

    # Authenticated users (#54) also get a long-term preferences blob updated
    # in this same LLM call — no separate extraction call/schedule needed.
    # Guests (user_id is None) keep the exact prior behaviour: summary only.
    if user_id is not None and pg_session is not None:
        existing_preferences = await load_preferences(pg_session, user_id)
        preferences_context = f"Known preferences so far: {existing_preferences}\n\n" if existing_preferences else ""
        prompt = (
            f"{prior_context}{preferences_context}"
            f"Given the conversation below, produce:\n"
            f"1. A 3-5 sentence summary of the conversation, from the assistant's perspective, "
            f"focusing on what the user was looking for and any products discussed.\n"
            f"2. Updated durable customer preferences (merge with the known preferences above, "
            f"if any) — things likely to still be true in a future, unrelated conversation, "
            f"such as preferred brands, price sensitivity, or categories of interest.\n\n{history_text}"
        )

        structured_llm = llm.with_structured_output(_SummaryAndPreferences)
        result = await structured_llm.ainvoke(prompt)
        summary = result.summary
        logger.info("Summary generated for conversation '%s': %s", session_id, summary[:80])

        await save_preferences(pg_session, user_id, result.preferences)
    else:
        prompt = (
            f"{prior_context}"
            f"Summarise the following conversation in 3-5 sentences. "
            f"Focus on what the user was looking for, any products discussed, and any preferences expressed. "
            f"Write from the assistant's perspective.\n\n{history_text}"
        )

        response = await llm.ainvoke(prompt)
        summary = response.content
        logger.info("Summary generated for conversation '%s': %s", session_id, summary[:80])

    await save_summary(session_id, summary)
    await save_messages(session_id, recent)

    return summary, recent
