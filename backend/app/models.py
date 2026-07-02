from pydantic import BaseModel
from langchain_core.chat_history import InMemoryChatMessageHistory
from typing import Optional, TypedDict


class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    message_count: int
    trace_id: str | None = None


class ConversationData(TypedDict):
    history: InMemoryChatMessageHistory
    is_new: bool

class FeedbackRequest(BaseModel):
    trace_id: str
    value: bool          # True = thumbs up, False = thumbs down
    comment: str | None = None


class FeedbackResponse(BaseModel):
    message: str