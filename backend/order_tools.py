import json
import logging
from typing import Annotated

from langchain_core.tools import InjectedToolArg, StructuredTool
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from db import get_session
from models_db import Order, OrderItem, Product
from session_identity import get_or_create_shadow_user

logger = logging.getLogger(__name__)


async def _order_summary(session, order: Order) -> dict:
    result = await session.execute(
        select(OrderItem.id, Product.name)
        .join(Product, OrderItem.product_id == Product.id)
        .where(OrderItem.order_id == order.id)
    )
    names_by_item_id = {item_id: name for item_id, name in result.all()}

    return {
        "id": order.id,
        "status": order.status,
        "created_at": order.created_at.isoformat(),
        "items": [
            {
                "product_id": i.product_id,
                "name": names_by_item_id.get(i.id, i.product_id),
                "quantity": i.quantity,
                "unit_price": i.unit_price,
            }
            for i in order.items
        ],
        "payment_status": order.payment.status if order.payment else "unknown",
        "amount": order.payment.amount if order.payment else None,
    }


async def view_order_history_impl(*, session_id: Annotated[str, InjectedToolArg]) -> str:
    try:
        async with get_session() as session:
            user = await get_or_create_shadow_user(session, session_id)

            result = await session.execute(
                select(Order)
                .where(Order.user_id == user.id)
                .options(selectinload(Order.items), selectinload(Order.payment))
                .order_by(Order.id.desc())
            )
            orders = result.scalars().all()

            summaries = [await _order_summary(session, o) for o in orders]
            await session.commit()

        return json.dumps({"orders": summaries})
    except Exception as e:
        logger.exception("Exception in view_order_history_impl: %s", str(e))
        return json.dumps({"error": str(e)})


_view_order_history_tool = StructuredTool.from_function(
    coroutine=view_order_history_impl,
    name="view_order_history",
    description=(
        "View the customer's past orders: order id, status, order date, line items "
        "(product name, quantity, unit price), and payment status/amount for each. "
        "Use when the customer asks what they've ordered before, wants to check on a "
        "past order, or asks about order/payment status for something already checked "
        "out. Do NOT use this for the customer's current, not-yet-checked-out cart — "
        "use view_cart for that instead."
    ),
)
# args_schema inferred from the function signature, not an explicit empty
# model — see cart_tools.py's _view_cart_tool for why (StructuredTool's
# zero-fields fast path silently drops the injected session_id otherwise).
_view_order_history_tool._needs_session_id = True

ORDER_TOOLS = [_view_order_history_tool]
