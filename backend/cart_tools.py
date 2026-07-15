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


class RemoveFromCartInput(BaseModel):
    product_id: str = Field(description="The id of the product to remove from the cart")


class UpdateQuantityInput(BaseModel):
    product_id: str = Field(description="The id of the product whose quantity should be updated")
    quantity: int = Field(description="The new total quantity for this item (not a delta)")


class GetProductInput(BaseModel):
    product_id: str = Field(description="The id of the product to look up")


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


async def remove_from_cart_impl(product_id: str, *, session_id: str) -> str:
    try:
        async with get_session() as session:
            user = await get_or_create_shadow_user(session, session_id)
            cart = await _get_or_create_cart(session, user.id)

            result = await session.execute(
                select(CartItem).where(CartItem.cart_id == cart.id, CartItem.product_id == product_id)
            )
            existing = result.scalar_one_or_none()
            if existing is None:
                return json.dumps({"error": "Item not in cart"})

            await session.delete(existing)
            await session.commit()

            summary = await _cart_summary(session, user.id)
            await session.commit()

        return json.dumps({"message": "Removed item from cart", **summary})
    except Exception as e:
        logger.exception("Exception in remove_from_cart_impl: %s", str(e))
        return json.dumps({"error": str(e)})


async def update_quantity_impl(product_id: str, quantity: int, *, session_id: str) -> str:
    try:
        if quantity <= 0:
            return json.dumps({"error": "Quantity must be a positive number"})

        async with get_session() as session:
            user = await get_or_create_shadow_user(session, session_id)
            cart = await _get_or_create_cart(session, user.id)

            result = await session.execute(
                select(CartItem).where(CartItem.cart_id == cart.id, CartItem.product_id == product_id)
            )
            existing = result.scalar_one_or_none()
            if existing is None:
                return json.dumps({"error": "Item not in cart"})

            existing.quantity = quantity
            await session.commit()

            summary = await _cart_summary(session, user.id)
            await session.commit()

        return json.dumps({"message": f"Updated quantity to {quantity}", **summary})
    except Exception as e:
        logger.exception("Exception in update_quantity_impl: %s", str(e))
        return json.dumps({"error": str(e)})


async def get_product_impl(product_id: str) -> str:
    try:
        async with get_session() as session:
            product = await session.get(Product, product_id)
            if product is None:
                return json.dumps({"error": "Product not found"})

            summary = json.dumps({
                "id": product.id,
                "name": product.name,
                "description": product.description,
                "category": product.category,
                "price": product.price,
                "original_price": product.original_price,
                "rating": product.rating,
                "reviews": product.reviews or 0,
                "image": product.image,
            })
            await session.commit()
        return summary
    except Exception as e:
        logger.exception("Exception in get_product_impl: %s", str(e))
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

_remove_from_cart_tool = StructuredTool(
    name="remove_from_cart",
    coroutine=remove_from_cart_impl,
    args_schema=RemoveFromCartInput,
    description=(
        "Remove a product entirely from the customer's cart, regardless of its current quantity. "
        "Use when the customer wants an item taken out of their cart completely (e.g. 'remove the widget', "
        "'take that off my cart', 'I don't want the X anymore'). "
        "Do NOT use this to reduce how many units of an item are in the cart — use update_quantity for that instead."
    ),
)
_remove_from_cart_tool._needs_session_id = True

_update_quantity_tool = StructuredTool(
    name="update_quantity",
    coroutine=update_quantity_impl,
    args_schema=UpdateQuantityInput,
    description=(
        "Change the quantity of a product already in the customer's cart to a new total amount "
        "(this sets the absolute quantity, it does not add or subtract units). "
        "Use when the customer wants more or fewer units of something already in their cart "
        "(e.g. 'make that 3 instead', 'I only want 1 now', 'change quantity to 5'). "
        "Quantity must be a positive number — to remove the item entirely, use remove_from_cart instead."
    ),
)
_update_quantity_tool._needs_session_id = True

_get_product_tool = StructuredTool(
    name="get_product",
    coroutine=get_product_impl,
    args_schema=GetProductInput,
    description=(
        "Look up full details (description, price, rating, category, etc.) for a single specific product "
        "by its id. Use when the customer asks for more details about one particular product you already "
        "know the id of (e.g. from a prior search result), not for browsing or searching the catalog."
    ),
)
# No _needs_session_id — product lookups are not session-scoped.

CART_TOOLS = [_add_to_cart_tool, _view_cart_tool, _remove_from_cart_tool, _update_quantity_tool, _get_product_tool]
