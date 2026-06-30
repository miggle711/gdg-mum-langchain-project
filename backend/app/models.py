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


class ConversationData(TypedDict):
    history: InMemoryChatMessageHistory
    is_new: bool
