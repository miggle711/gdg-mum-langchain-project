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

from conversations import save_messages, load_messages, maybe_summarise
from cache import _get_redis
from app.models import ChatRequest, ChatResponse, ConversationData, FeedbackRequest, FeedbackResponse
from app.agent import _llm, langfuse_client
from app.graph import chat_graph
from app.limiter import limiter

router = APIRouter()


def get_or_create_conversation(session_id: str) -> ConversationData:
    messages = load_messages(session_id)
    history = InMemoryChatMessageHistory()
    history.add_messages(messages)
    return {
        "history": history,
    }


@router.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    try:
        conversation = get_or_create_conversation(body.session_id)
        history = conversation["history"]

        summary, recent_messages = maybe_summarise(body.session_id, history.messages, _llm)

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
                }
            )

        history.add_user_message(body.message)
        history.add_ai_message(response_text)
        save_messages(body.session_id, history.messages)

        return ChatResponse(
            session_id=body.session_id,
            response=response_text or "I apologize, but I'm having trouble generating a response at the moment.",
            message_count=len(history.messages) // 2,
            trace_id=trace_id,
        )
    except Exception as e:
        logger.exception("Exception in chat: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    conversation = get_or_create_conversation(body.session_id)
    history = conversation["history"]

    summary, recent_messages = maybe_summarise(body.session_id, history.messages, _llm)

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
                full_response = response_text

                span.update(
                    output={
                        "intent": result.get("intent"),
                        "response": response_text,
                    }
                )

                yield f"data: {json.dumps({'text': response_text})}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        finally:
            if full_response:
                history.add_user_message(body.message)
                history.add_ai_message(full_response)
                save_messages(body.session_id, history.messages)

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/session/start")
def start_session() -> dict[str, str]:
    session_id = str(uuid.uuid4())
    get_or_create_conversation(session_id)
    return {
        "session_id": session_id,
        "message": "Welcome to our store! How can I help you find the perfect product today?"
    }


@router.get("/conversation/{session_id}")
def get_conversation(session_id: str):
    messages = load_messages(session_id)
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
def delete_conversation(session_id: str):
    r = _get_redis()
    deleted = r.delete(f"conversation:{session_id}")
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted"}


@router.get("/conversations")
def list_conversations():
    r = _get_redis()
    keys = r.keys("conversation:*")
    return {
        "conversations": [
            {
                "session_id": key.split(":", 1)[1],
                "message_count": len(load_messages(key.split(":", 1)[1])) // 2,
            }
            for key in keys
        ]
    }

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
