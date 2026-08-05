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

from auth import decode_access_token
from conversations import save_messages, load_messages, maybe_summarise
from cache import _get_redis
from app.models import ChatRequest, ChatResponse, ConversationData, FeedbackRequest, FeedbackResponse
from app.agent import _llm, langfuse_client
from app.graph import chat_graph
from app.limiter import limiter

router = APIRouter()


def _conversation_key(request: Request, session_id: str) -> str:
    """Keys conversation history by user_id when a valid JWT is present,
    falling back to session_id for guests (#82) — so a logged-in user's
    conversation history can't be read/deleted by anyone who guesses or is
    handed their session_id, since a real user_id can't be forged without
    the JWT secret. Deliberately does NOT call resolve_user/hit Postgres:
    a JWT's signature alone proves the user_id claim, no DB lookup needed
    just to pick a Redis key, and this keeps /chat's guest path exactly as
    dependency-free as it is today (no forced shadow-user DB write per
    anonymous message).
    """
    auth_header = request.headers.get("authorization", "")
    token = auth_header.removeprefix("Bearer ").strip()
    user_id = decode_access_token(token) if token else None
    return str(user_id) if user_id is not None else session_id


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
        conversation_key = _conversation_key(request, body.session_id)
        conversation = await get_or_create_conversation(conversation_key)
        history = conversation["history"]

        summary, recent_messages = await maybe_summarise(conversation_key, history.messages, _llm)

        chat_history = []
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
    conversation_key = _conversation_key(request, body.session_id)
    conversation = await get_or_create_conversation(conversation_key)
    history = conversation["history"]

    summary, recent_messages = await maybe_summarise(conversation_key, history.messages, _llm)

    chat_history = []
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
    conversation_key = _conversation_key(request, session_id)
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
    conversation_key = _conversation_key(request, session_id)
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
