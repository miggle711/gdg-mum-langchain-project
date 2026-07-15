import json
import logging

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy import select

from db import get_session
from models_db import Cart, CartItem, Product
from session_identity import get_or_create_shadow_user

logger = logging.getLogger(__name__)


class AddToCartInput(BaseModel):
    product_id: str = Field(description="The id of the product to add to the cart")
    quantity: int = Field(default=1, description="How many units to add")


class ViewCartInput(BaseModel):
    pass


async def _get_or_create_cart(session, user_id: int) -> Cart:
    result = await session.execute(select(Cart).where(Cart.user_id == user_id))
    cart = result.scalar_one_or_none()
    if cart is None:
        cart = Cart(user_id=user_id)
        session.add(cart)
        await session.flush()
    return cart


async def _cart_summary(session, user_id: int) -> dict:
    cart = await _get_or_create_cart(session, user_id)
    result = await session.execute(
        select(CartItem, Product.name, Product.price)
        .join(Product, CartItem.product_id == Product.id)
        .where(CartItem.cart_id == cart.id)
    )
    rows = result.all()
    items = [
        {"product_id": ci.product_id, "name": name, "quantity": ci.quantity}
        for ci, name, _ in rows
    ]
    total = sum(price * ci.quantity for ci, _, price in rows)
    return {"items": items, "total": round(total, 2)}


async def add_to_cart_impl(product_id: str, quantity: int = 1, *, session_id: str) -> str:
    try:
        if quantity <= 0:
            return json.dumps({"error": "Quantity must be a positive number"})

        async with get_session() as session:
            product = await session.get(Product, product_id)
            if product is None:
                return json.dumps({"error": "Product not found"})

            user = await get_or_create_shadow_user(session, session_id)
            cart = await _get_or_create_cart(session, user.id)

            result = await session.execute(
                select(CartItem).where(CartItem.cart_id == cart.id, CartItem.product_id == product_id)
            )
            existing = result.scalar_one_or_none()
            if existing:
                existing.quantity += quantity
            else:
                session.add(CartItem(cart_id=cart.id, product_id=product_id, quantity=quantity))
            await session.commit()

            summary = await _cart_summary(session, user.id)
            await session.commit()

        return json.dumps({"message": f"Added {quantity} x {product.name} to cart", **summary})
    except Exception as e:
        logger.exception("Exception in add_to_cart_impl: %s", str(e))
        return json.dumps({"error": str(e)})


async def view_cart_impl(*, session_id: str) -> str:
    try:
        async with get_session() as session:
            user = await get_or_create_shadow_user(session, session_id)
            summary = await _cart_summary(session, user.id)
            await session.commit()
        return json.dumps(summary)
    except Exception as e:
        logger.exception("Exception in view_cart_impl: %s", str(e))
        return json.dumps({"error": str(e)})


_add_to_cart_tool = StructuredTool(
    name="add_to_cart",
    coroutine=add_to_cart_impl,
    args_schema=AddToCartInput,
    description="Add a product to the customer's cart. Use when the customer asks to add, buy, or order a specific product.",
)
_add_to_cart_tool._needs_session_id = True

_view_cart_tool = StructuredTool(
    name="view_cart",
    coroutine=view_cart_impl,
    args_schema=ViewCartInput,
    description="View the current contents and total of the customer's cart. Use when the customer asks what's in their cart.",
)
_view_cart_tool._needs_session_id = True

CART_TOOLS = [_add_to_cart_tool, _view_cart_tool]
