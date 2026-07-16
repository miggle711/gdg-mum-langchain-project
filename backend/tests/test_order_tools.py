"""Structural tests for the order-history tool (backend/order_tools.py, #33)
— exercised directly against the ORM via the same in-memory async SQLite
fixture pattern as test_cart_tools.py, not through LangChain's tool-calling
machinery (StructuredTool.ainvoke).
"""

import json

import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db import Base
from models_db import Address, Order, OrderItem, Payment, Product
from session_identity import get_or_create_shadow_user
import order_tools


@pytest_asyncio.fixture
async def session(mocker):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    mocker.patch("order_tools.get_session", side_effect=session_factory)

    async with session_factory() as s:
        yield s

    await engine.dispose()


async def _seed_order(
    session,
    session_id="session-1",
    product_id="p1",
    name="Widget",
    unit_price=9.99,
    quantity=2,
    status="paid",
    payment_status="succeeded",
):
    product = Product(id=product_id, name=name, category="Electronics", price=unit_price)
    session.add(product)
    user = await get_or_create_shadow_user(session, session_id)
    address = Address(user_id=user.id, street="1 Main St", city="Springfield", zip_code="00000", country="US")
    session.add(address)
    await session.flush()

    order = Order(user_id=user.id, address_id=address.id, status=status)
    session.add(order)
    await session.flush()

    session.add(OrderItem(order_id=order.id, product_id=product_id, quantity=quantity, unit_price=unit_price))
    session.add(Payment(
        order_id=order.id,
        amount=round(unit_price * quantity, 2),
        status=payment_status,
        provider_reference="mock_test",
    ))
    await session.commit()
    return order


async def test_view_order_history_impl_returns_empty_list_for_new_session(session):
    result = json.loads(await order_tools.view_order_history_impl(session_id="brand-new-session"))

    assert result["orders"] == []


async def test_view_order_history_impl_returns_order_with_items_and_payment(session):
    await _seed_order(session, session_id="session-1")

    result = json.loads(await order_tools.view_order_history_impl(session_id="session-1"))

    assert len(result["orders"]) == 1
    order = result["orders"][0]
    assert order["status"] == "paid"
    assert order["payment_status"] == "succeeded"
    assert order["amount"] == 19.98
    assert order["items"] == [{"product_id": "p1", "name": "Widget", "quantity": 2, "unit_price": 9.99}]
    assert "created_at" in order


async def test_view_order_history_impl_only_returns_orders_for_this_session(session):
    await _seed_order(session, session_id="session-1", product_id="p1")
    await _seed_order(session, session_id="session-2", product_id="p2")

    result = json.loads(await order_tools.view_order_history_impl(session_id="session-1"))

    assert len(result["orders"]) == 1
    assert result["orders"][0]["items"][0]["product_id"] == "p1"


async def test_view_order_history_impl_orders_most_recent_first(session):
    order1 = await _seed_order(session, session_id="session-1", product_id="p1", name="First")
    order2 = await _seed_order(session, session_id="session-1", product_id="p2", name="Second")

    result = json.loads(await order_tools.view_order_history_impl(session_id="session-1"))

    assert [o["id"] for o in result["orders"]] == [order2.id, order1.id]


async def test_view_order_history_impl_handles_missing_payment_gracefully(session):
    product = Product(id="p1", name="Widget", category="Electronics", price=9.99)
    session.add(product)
    user = await get_or_create_shadow_user(session, "session-1")
    address = Address(user_id=user.id, street="1 Main St", city="Springfield", zip_code="00000", country="US")
    session.add(address)
    await session.flush()

    order = Order(user_id=user.id, address_id=address.id, status="pending")
    session.add(order)
    await session.flush()
    session.add(OrderItem(order_id=order.id, product_id="p1", quantity=1, unit_price=9.99))
    await session.commit()  # no Payment row created

    result = json.loads(await order_tools.view_order_history_impl(session_id="session-1"))

    assert result["orders"][0]["payment_status"] == "unknown"
    assert result["orders"][0]["amount"] is None
