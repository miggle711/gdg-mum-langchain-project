import uuid
import logging
import sys
import os

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from db import get_session
from models_db import Address, Cart, CartItem, Order, OrderItem, Payment, Product
from session_identity import get_or_create_shadow_user
from app.models import (
    AddToCartRequest,
    CartItemResponse,
    CartResponse,
    CheckoutRequest,
    CheckoutResponse,
    OrderItemResponse,
    OrderResponse,
    UpdateCartItemRequest,
)

router = APIRouter()


async def _get_or_create_cart(session, user_id: int) -> Cart:
    result = await session.execute(select(Cart).where(Cart.user_id == user_id))
    cart = result.scalar_one_or_none()
    if cart is None:
        cart = Cart(user_id=user_id)
        session.add(cart)
        await session.flush()  # populate cart.id without committing yet
    return cart


async def _load_cart_response(session, user_id: int, session_id: str) -> CartResponse:
    cart = await _get_or_create_cart(session, user_id)
    result = await session.execute(
        select(CartItem, Product.price)
        .join(Product, CartItem.product_id == Product.id)
        .where(CartItem.cart_id == cart.id)
    )
    rows = result.all()
    items = [CartItemResponse(id=ci.id, product_id=ci.product_id, quantity=ci.quantity) for ci, _ in rows]
    total = sum(price * ci.quantity for ci, price in rows)
    return CartResponse(session_id=session_id, items=items, total=round(total, 2))


@router.get("/cart/{session_id}")
async def get_cart(session_id: str) -> CartResponse:
    try:
        async with get_session() as session:
            user = await get_or_create_shadow_user(session, session_id)
            response = await _load_cart_response(session, user.id, session_id)
            await session.commit()
        return response
    except Exception as e:
        logger.exception("Exception in get_cart: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cart/add")
async def add_to_cart(body: AddToCartRequest) -> CartResponse:
    try:
        async with get_session() as session:
            product = await session.get(Product, body.product_id)
            if product is None:
                raise HTTPException(status_code=404, detail="Product not found")

            user = await get_or_create_shadow_user(session, body.session_id)
            cart = await _get_or_create_cart(session, user.id)

            result = await session.execute(
                select(CartItem).where(CartItem.cart_id == cart.id, CartItem.product_id == body.product_id)
            )
            existing = result.scalar_one_or_none()
            if existing:
                existing.quantity += body.quantity
            else:
                session.add(CartItem(cart_id=cart.id, product_id=body.product_id, quantity=body.quantity))
            await session.commit()

            response = await _load_cart_response(session, user.id, body.session_id)
            await session.commit()
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Exception in add_to_cart: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/cart/item/{item_id}")
async def update_cart_item(item_id: int, body: UpdateCartItemRequest) -> dict[str, str]:
    try:
        async with get_session() as session:
            item = await session.get(CartItem, item_id)
            if item is None:
                raise HTTPException(status_code=404, detail="Cart item not found")
            if body.quantity <= 0:
                await session.delete(item)
            else:
                item.quantity = body.quantity
            await session.commit()
        return {"message": "Cart item updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Exception in update_cart_item: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cart/item/{item_id}")
async def remove_cart_item(item_id: int) -> dict[str, str]:
    try:
        async with get_session() as session:
            item = await session.get(CartItem, item_id)
            if item is None:
                raise HTTPException(status_code=404, detail="Cart item not found")
            await session.delete(item)
            await session.commit()
        return {"message": "Cart item removed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Exception in remove_cart_item: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/checkout")
async def checkout(body: CheckoutRequest) -> CheckoutResponse:
    try:
        async with get_session() as session:
            user = await get_or_create_shadow_user(session, body.session_id)

            # No self-serve address management exists yet for session-only
            # (shadow) users — checkout requires an address already owned
            # by this session's user. This also closes a pre-existing gap
            # where any client could pass any user_id+address_id pair with
            # no ownership check at all.
            address_result = await session.execute(
                select(Address).where(Address.id == body.address_id, Address.user_id == user.id)
            )
            if address_result.scalar_one_or_none() is None:
                raise HTTPException(status_code=400, detail="Address not found for this session")

            async with session.begin():
                cart_result = await session.execute(select(Cart).where(Cart.user_id == user.id))
                cart = cart_result.scalar_one_or_none()
                if cart is None:
                    raise HTTPException(status_code=400, detail="Cart is empty")

                items_result = await session.execute(
                    select(CartItem, Product.price)
                    .join(Product, CartItem.product_id == Product.id)
                    .where(CartItem.cart_id == cart.id)
                )
                rows = items_result.all()
                if not rows:
                    raise HTTPException(status_code=400, detail="Cart is empty")

                order = Order(user_id=user.id, address_id=body.address_id, status="paid")
                session.add(order)
                await session.flush()

                total = 0.0
                for cart_item, price in rows:
                    session.add(OrderItem(
                        order_id=order.id,
                        product_id=cart_item.product_id,
                        quantity=cart_item.quantity,
                        unit_price=price,
                    ))
                    total += price * cart_item.quantity

                session.add(Payment(
                    order_id=order.id,
                    amount=round(total, 2),
                    status="succeeded",
                    provider_reference=f"mock_{uuid.uuid4().hex}",
                ))

                for cart_item, _ in rows:
                    await session.delete(cart_item)
                # session.begin() block commits on clean exit here, or rolls
                # back the entire transaction if anything above raised —
                # guarantees no order without a payment, no cleared cart
                # without a completed order.

            order_id = order.id  # stays populated post-commit (expire_on_commit=False in db.py)
            reload_result = await session.execute(
                select(Order)
                .where(Order.id == order_id)
                .options(selectinload(Order.items), selectinload(Order.payment))
            )
            fresh_order = reload_result.scalar_one()

            response = CheckoutResponse(
                order=OrderResponse(
                    id=fresh_order.id,
                    session_id=body.session_id,
                    address_id=fresh_order.address_id,
                    status=fresh_order.status,
                    items=[
                        OrderItemResponse(product_id=i.product_id, quantity=i.quantity, unit_price=i.unit_price)
                        for i in fresh_order.items
                    ],
                    payment_status=fresh_order.payment.status,
                ),
                message="Order placed successfully",
            )
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Exception in checkout: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders/{session_id}")
async def list_orders(session_id: str) -> list[OrderResponse]:
    try:
        async with get_session() as session:
            user = await get_or_create_shadow_user(session, session_id)
            await session.commit()
            result = await session.execute(
                select(Order)
                .where(Order.user_id == user.id)
                .options(selectinload(Order.items), selectinload(Order.payment))
            )
            orders = result.scalars().all()
            return [
                OrderResponse(
                    id=o.id,
                    session_id=session_id,
                    address_id=o.address_id,
                    status=o.status,
                    items=[
                        OrderItemResponse(product_id=i.product_id, quantity=i.quantity, unit_price=i.unit_price)
                        for i in o.items
                    ],
                    payment_status=o.payment.status if o.payment else "unknown",
                )
                for o in orders
            ]
    except Exception as e:
        logger.exception("Exception in list_orders: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
