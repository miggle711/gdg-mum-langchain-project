import uuid
import json
import asyncio
import logging

logger = logging.getLogger(__name__)
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
# from langfuse import propagate_attributes
from langfuse.langchain import CallbackHandler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from auth import decode_access_token, extract_bearer_token
from conversations import save_messages, load_messages, load_preferences, maybe_summarise
from cache import _get_redis
from db import get_session
from app.models import ChatRequest, ChatResponse, ConversationData, FeedbackRequest, FeedbackResponse
from app.agent import _llm, langfuse_client
from app.graph import chat_graph
from app.limiter import limiter

router = APIRouter()


def _authenticated_user_id(request: Request) -> int | None:
    """Verifies the request's JWT (if any) and returns the claimed user_id,
    or None for guests/invalid tokens. A JWT's signature alone proves the
    user_id claim — no DB lookup needed just to know who's asking (#82)."""
    token = extract_bearer_token(request.headers.get("authorization"))
    return decode_access_token(token) if token else None


_AUTHENTICATED_KEY_PREFIX = "user:"


def _conversation_key(session_id: str, user_id: int | None) -> str:
    """Keys conversation history by user_id when authenticated, falling back
    to session_id for guests (#82) — so a logged-in user's conversation
    history can't be read/deleted by anyone who guesses or is handed their
    session_id. Deliberately does NOT hit Postgres itself: this keeps /chat's
    guest path exactly as dependency-free as it is today (no forced
    shadow-user DB write per anonymous message).

    The authenticated key is prefixed ("user:{id}") rather than being the
    bare numeric user_id — a guest client could otherwise send
    session_id="55" and land on the exact same Redis key as user_id=55's
    authenticated conversation. The prefix makes the two namespaces
    structurally disjoint regardless of what a guest sends, rather than
    relying on session_id's UUID shape as an (unenforced) assumption.

    A guest session_id that itself starts with the reserved "user:" prefix
    (e.g. a malicious client sending session_id="user:55") is re-namespaced
    under "guest:" rather than trusted verbatim — otherwise a guest could
    directly spoof an authenticated user's key and read/write their real
    conversation history with no valid JWT at all.
    """
    if user_id is not None:
        return f"{_AUTHENTICATED_KEY_PREFIX}{user_id}"
    if session_id.startswith(_AUTHENTICATED_KEY_PREFIX):
        return f"guest:{session_id}"
    return session_id


async def get_or_create_conversation(session_id: str) -> ConversationData:
    messages = await load_messages(session_id)
    history = InMemoryChatMessageHistory()
    history.add_messages(messages)
    return {
        "history": history,
    }


@router.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    try:
        user_id = _authenticated_user_id(request)
        conversation_key = _conversation_key(body.session_id, user_id)
        conversation = await get_or_create_conversation(conversation_key)
        history = conversation["history"]

        chat_history = []

        # Long-term memory (#54): authenticated users only — guests never
        # touch Postgres on the chat path (#82's guest-path guarantee).
        if user_id is not None:
            async with get_session() as pg_session:
                preferences = await load_preferences(pg_session, user_id)
                if preferences:
                    chat_history.append(SystemMessage(content=f"What we know about this customer: {preferences}"))
                summary, recent_messages = await maybe_summarise(
                    conversation_key, history.messages, _llm, pg_session=pg_session, user_id=user_id
                )
        else:
            summary, recent_messages = await maybe_summarise(conversation_key, history.messages, _llm)

        if summary:
            chat_history.append(SystemMessage(content=f"Summary of earlier conversation: {summary}"))
        chat_history.extend(recent_messages)

        trace_id = langfuse_client.create_trace_id()
        handler = CallbackHandler(trace_context={"trace_id": trace_id})

        with langfuse_client.start_as_current_span(
            name="http.chat",
            trace_context={"trace_id": trace_id},
            input={
                "session_id": body.session_id,
                "message": body.message,
                "chat_history_length": len(chat_history),
            },
            metadata={"route": "/chat"},
        ) as span:
            langfuse_client.update_current_trace(
                name="ecommerce-chat",
                session_id=body.session_id,
                input={"message": body.message},
                metadata={"route": "/chat"},
            )

            result = await chat_graph.ainvoke(
                {"input": body.message, "chat_history": chat_history, "session_id": body.session_id},
                config={"callbacks": [handler]},
            )

            response_text = result["response"]

            span.update(
                output={
                    "intent": result.get("intent"),
                    "response": response_text,
                    "error": result.get("error", False),
                }
            )

        if result.get("error"):
            # Intent classification failed (e.g. quota exhaustion, timeout) —
            # surface this as a real failure instead of returning a 200 with
            # clarify_node's placeholder text indistinguishable from a
            # genuine successful reply (#77).
            raise HTTPException(status_code=502, detail="Failed to process message, please try again.")

        history.add_user_message(body.message)
        history.add_ai_message(response_text)
        await save_messages(conversation_key, history.messages)

        return ChatResponse(
            session_id=body.session_id,
            response=response_text or "I apologize, but I'm having trouble generating a response at the moment.",
            message_count=len(history.messages) // 2,
            trace_id=trace_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Exception in chat: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    user_id = _authenticated_user_id(request)
    conversation_key = _conversation_key(body.session_id, user_id)
    conversation = await get_or_create_conversation(conversation_key)
    history = conversation["history"]

    chat_history = []

    # Long-term memory (#54): authenticated users only — guests never touch
    # Postgres on the chat path (#82's guest-path guarantee).
    if user_id is not None:
        async with get_session() as pg_session:
            preferences = await load_preferences(pg_session, user_id)
            if preferences:
                chat_history.append(SystemMessage(content=f"What we know about this customer: {preferences}"))
            summary, recent_messages = await maybe_summarise(
                conversation_key, history.messages, _llm, pg_session=pg_session, user_id=user_id
            )
    else:
        summary, recent_messages = await maybe_summarise(conversation_key, history.messages, _llm)

    if summary:
        chat_history.append(SystemMessage(content=f"Summary of earlier conversation: {summary}"))
    chat_history.extend(recent_messages)

    # Generate the trace_id upfront so we can send it to the frontend
    # before any tokens stream back — it needs to be attached to this
    # specific message for the feedback buttons.
    trace_id = langfuse_client.create_trace_id()
    stream_handler = CallbackHandler(trace_context={"trace_id": trace_id})

    async def generate():
        full_response = ""
        yield f"data: {json.dumps({'trace_id': trace_id})}\n\n"

        try:
            with langfuse_client.start_as_current_span(
                name="http.chat_stream",
                trace_context={"trace_id": trace_id},
                input={
                    "session_id": body.session_id,
                    "message": body.message,
                    "chat_history_length": len(chat_history),
                },
                metadata={"route": "/chat/stream"},
            ) as span:
                langfuse_client.update_current_trace(
                    name="ecommerce-chat-stream",
                    session_id=body.session_id,
                    input={"message": body.message},
                    metadata={"route": "/chat/stream"},
                )

                result = await chat_graph.ainvoke(
                    {"input": body.message, "chat_history": chat_history, "session_id": body.session_id},
                    config={"callbacks": [stream_handler]},
                )
                response_text = result["response"]

                span.update(
                    output={
                        "intent": result.get("intent"),
                        "response": response_text,
                        "error": result.get("error", False),
                    }
                )

                if result.get("error"):
                    # Same rationale as /chat (#77): don't stream back
                    # clarify_node's placeholder text as if it were a normal
                    # successful reply when classification actually failed.
                    yield f"data: {json.dumps({'error': 'Failed to process message, please try again.'})}\n\n"
                    return

                full_response = response_text
                yield f"data: {json.dumps({'text': response_text})}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        finally:
            if full_response:
                history.add_user_message(body.message)
                history.add_ai_message(full_response)
                await save_messages(conversation_key, history.messages)

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/session/start")
async def start_session() -> dict[str, str]:
    session_id = str(uuid.uuid4())
    await get_or_create_conversation(session_id)
    return {
        "session_id": session_id,
        "message": "Welcome to our store! How can I help you find the perfect product today?"
    }


@router.get("/conversation/{session_id}")
async def get_conversation(request: Request, session_id: str):
    conversation_key = _conversation_key(session_id, _authenticated_user_id(request))
    messages = await load_messages(conversation_key)
    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages_out = []
    for msg in messages:
        role = "human" if isinstance(msg, HumanMessage) else "ai"
        messages_out.append({"role": role, "content": msg.content})

    return {
        "session_id": session_id,
        "history": messages_out,
        "message_count": len(messages) // 2,
    }


@router.delete("/conversation/{session_id}")
async def delete_conversation(request: Request, session_id: str):
    conversation_key = _conversation_key(session_id, _authenticated_user_id(request))
    r = _get_redis()
    deleted = await r.delete(f"conversation:{conversation_key}")
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted"}


@router.get("/conversations")
async def list_conversations():
    r = _get_redis()
    keys = await r.keys("conversation:*")
    conversations = []
    for key in keys:
        session_id = key.split(":", 1)[1]
        messages = await load_messages(session_id)
        conversations.append({
            "session_id": session_id,
            "message_count": len(messages) // 2,
        })
    return {"conversations": conversations}

@router.post("/feedback")
def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    try:
        langfuse_client.create_score(
            trace_id=body.trace_id,
            name="user-feedback",
            value=body.value,
            data_type="BOOLEAN",
            comment=body.comment,
        )
        return FeedbackResponse(message="Feedback recorded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
