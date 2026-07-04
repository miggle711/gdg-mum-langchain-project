import uuid
import json
import asyncio
import logging

logger = logging.getLogger(__name__)
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import propagate_attributes
from langfuse.langchain import CallbackHandler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conversations import save_messages, load_messages, maybe_summarise
from cache import _get_redis
from app.models import ChatRequest, ChatResponse, ConversationData, FeedbackRequest, FeedbackResponse
from app.agent import agent_executor, _llm, langfuse_client
from app.limiter import limiter

router = APIRouter()


def get_or_create_conversation(conversation_id: str) -> ConversationData:
    messages = load_messages(conversation_id)
    history = InMemoryChatMessageHistory()
    history.add_messages(messages)
    return {
        "history": history,
        "is_new": len(messages) == 0,
    }


@router.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    try:
        conversation = get_or_create_conversation(body.conversation_id)
        history = conversation["history"]

        summary, recent_messages = maybe_summarise(body.conversation_id, history.messages, _llm)

        chat_history = []
        if summary:
            chat_history.append(SystemMessage(content=f"Summary of earlier conversation: {summary}"))
        chat_history.extend(recent_messages)

        trace_id = langfuse_client.create_trace_id()
        handler = CallbackHandler(trace_context={"trace_id": trace_id})
        result = agent_executor.invoke(
            {"input": body.message, "chat_history": chat_history},
            config={"callbacks": [handler]},
        )

        response_text = result["output"]

        history.add_user_message(body.message)
        history.add_ai_message(response_text)
        save_messages(body.conversation_id, history.messages)

        return ChatResponse(
            conversation_id=body.conversation_id,
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
    conversation = get_or_create_conversation(body.conversation_id)
    history = conversation["history"]

    summary, recent_messages = maybe_summarise(body.conversation_id, history.messages, _llm)

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
            with propagate_attributes(session_id=body.conversation_id, trace_name="ecommerce-chat-stream"):
                async for chunk in agent_executor.astream(
                    {"input": body.message, "chat_history": chat_history},
                    config={"callbacks": [stream_handler]},
                ):
                    if "output" in chunk:
                        token = chunk["output"]
                        full_response += token
                        yield f"data: {json.dumps({'text': token})}\n\n"
                        await asyncio.sleep(0)  # yield control back to the event loop
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        finally:
            if full_response:
                history.add_user_message(body.message)
                history.add_ai_message(full_response)
                save_messages(body.conversation_id, history.messages)

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/chat/start")
def start_conversation() -> dict[str, str]:
    conversation_id = str(uuid.uuid4())
    get_or_create_conversation(conversation_id)
    return {
        "conversation_id": conversation_id,
        "message": "Welcome to our store! How can I help you find the perfect product today?"
    }


@router.get("/conversation/{conversation_id}")
def get_conversation(conversation_id: str):
    messages = load_messages(conversation_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages_out = []
    for msg in messages:
        role = "human" if isinstance(msg, HumanMessage) else "ai"
        messages_out.append({"role": role, "content": msg.content})

    return {
        "conversation_id": conversation_id,
        "history": messages_out,
        "message_count": len(messages) // 2,
    }


@router.delete("/conversation/{conversation_id}")
def delete_conversation(conversation_id: str):
    r = _get_redis()
    deleted = r.delete(f"conversation:{conversation_id}")
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
                "conversation_id": key.split(":", 1)[1],
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