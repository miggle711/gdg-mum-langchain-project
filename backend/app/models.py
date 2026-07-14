from pydantic import BaseModel
from langchain_core.chat_history import InMemoryChatMessageHistory
from typing import TypedDict


class ChatRequest(BaseModel):
    conversation_id: str
    message: str


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    message_count: int
    trace_id: str | None = None


class ConversationData(TypedDict):
    history: InMemoryChatMessageHistory

class FeedbackRequest(BaseModel):
    trace_id: str
    value: bool          # True = thumbs up, False = thumbs down
    comment: str | None = None


class FeedbackResponse(BaseModel):
    message: str


class CartItemResponse(BaseModel):
    id: int
    product_id: str
    quantity: int


class CartResponse(BaseModel):
    user_id: int
    items: list[CartItemResponse]
    total: float


class AddToCartRequest(BaseModel):
    user_id: int
    product_id: str
    quantity: int = 1


class UpdateCartItemRequest(BaseModel):
    quantity: int


class CheckoutRequest(BaseModel):
    user_id: int
    address_id: int


class OrderItemResponse(BaseModel):
    product_id: str
    quantity: int
    unit_price: float


class OrderResponse(BaseModel):
    id: int
    user_id: int
    address_id: int
    status: str
    items: list[OrderItemResponse]
    payment_status: str


class CheckoutResponse(BaseModel):
    order: OrderResponse
    message: str
