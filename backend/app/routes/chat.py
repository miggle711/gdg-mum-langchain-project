import uuid
import json
import asyncio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import propagate_attributes
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conversations import save_messages, load_messages, maybe_summarise
from cache import _get_redis
from app.models import ChatRequest, ChatResponse, ConversationData
from app.agent import agent_executor, _llm, langfuse_handler
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

        result = agent_executor.invoke({
            "input": body.message,
            "chat_history": chat_history,
        })
        response_text = result["output"]

        history.add_user_message(body.message)
        history.add_ai_message(response_text)
        save_messages(body.conversation_id, history.messages)

        return ChatResponse(
            conversation_id=body.conversation_id,
            response=response_text or "I apologize, but I'm having trouble generating a response at the moment.",
            message_count=len(history.messages) // 2,
        )
    except Exception as e:
        print(f"Exception in chat: {str(e)}")
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

    async def generate():
        full_response = ""
        try:
            async for chunk in agent_executor.astream(
                {"input": body.message, "chat_history": chat_history}
            ):
                # astream yields intermediate steps and the final output — we only want text chunks
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

# to group traces by conversation
with propagate_attributes(
    session_id=body.conversation_id,
    trace_name="ecommerce-chat",
):
    result = agent_executor.invoke(
        {
            "input": body.message,
            "chat_history": history.messages,
        },
        config={"callbacks": [langfuse_handler]},
    )