from pydantic import BaseModel
from langchain_core.chat_history import InMemoryChatMessageHistory
from typing import TypedDict


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
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
    session_id: str
    items: list[CartItemResponse]
    total: float


class AddToCartRequest(BaseModel):
    session_id: str
    product_id: str
    quantity: int = 1


class UpdateCartItemRequest(BaseModel):
    quantity: int


class CheckoutRequest(BaseModel):
    session_id: str
    address_id: int


class OrderItemResponse(BaseModel):
    product_id: str
    quantity: int
    unit_price: float


class OrderResponse(BaseModel):
    id: int
    session_id: str
    address_id: int
    status: str
    items: list[OrderItemResponse]
    payment_status: str


class CheckoutResponse(BaseModel):
    order: OrderResponse
    message: str


class ProductCreateRequest(BaseModel):
    name: str
    description: str | None = None
    category: str
    price: float
    original_price: float | None = None
    rating: float | None = None
    reviews: int = 0
    image: str | None = None


class ProductUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    category: str | None = None
    price: float | None = None
    original_price: float | None = None
    rating: float | None = None
    reviews: int | None = None
    image: str | None = None


class ProductResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    category: str
    price: float
    original_price: float | None = None
    rating: float | None = None
    reviews: int
    image: str | None = None
