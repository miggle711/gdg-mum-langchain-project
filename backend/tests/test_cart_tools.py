"""Structural tests for the cart-action tools (backend/cart_tools.py, #61)
— exercised directly against the ORM via the same in-memory async SQLite
fixture pattern as test_cart_routes.py, not through LangChain's tool-calling
machinery (StructuredTool.ainvoke). Tests call the _impl functions directly,
matching test_tools.py's existing convention.
"""

import json

import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db import Base
from models_db import Product
import cart_tools


@pytest_asyncio.fixture
async def session(mocker):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    # cart_tools' _impl functions open their own session via db.get_session()
    # (they have no caller-supplied session param, unlike cart.py's route
    # handlers) since a real tool call has no HTTP request session to reuse.
    # Patch get_session to hand out sessions from this test's in-memory engine.
    mocker.patch("cart_tools.get_session", side_effect=session_factory)

    async with session_factory() as s:
        yield s

    await engine.dispose()


async def _seed_product(session, product_id="p1", name="Widget", price=9.99):
    product = Product(id=product_id, name=name, category="Electronics", price=price)
    session.add(product)
    await session.commit()
    return product


async def test_add_to_cart_impl_creates_cart_item_for_new_shadow_user(session):
    await _seed_product(session)

    result = json.loads(await cart_tools.add_to_cart_impl(product_id="p1", quantity=2, session_id="session-1"))

    assert "error" not in result
    assert result["items"] == [{"product_id": "p1", "name": "Widget", "quantity": 2}]
    assert result["total"] == 19.98


async def test_add_to_cart_impl_increments_existing_item_quantity(session):
    await _seed_product(session)

    await cart_tools.add_to_cart_impl(product_id="p1", quantity=1, session_id="session-1")
    result = json.loads(await cart_tools.add_to_cart_impl(product_id="p1", quantity=2, session_id="session-1"))

    assert result["items"] == [{"product_id": "p1", "name": "Widget", "quantity": 3}]


async def test_add_to_cart_impl_returns_error_json_for_missing_product(session):
    result = json.loads(await cart_tools.add_to_cart_impl(product_id="nonexistent", quantity=1, session_id="session-1"))

    assert result["error"] == "Product not found"


async def test_add_to_cart_impl_rejects_non_positive_quantity(session):
    await _seed_product(session)

    result = json.loads(await cart_tools.add_to_cart_impl(product_id="p1", quantity=0, session_id="session-1"))

    assert "error" in result
    view = json.loads(await cart_tools.view_cart_impl(session_id="session-1"))
    assert view["items"] == []


async def test_view_cart_impl_returns_current_items_and_total(session):
    await _seed_product(session)
    await cart_tools.add_to_cart_impl(product_id="p1", quantity=2, session_id="session-1")

    result = json.loads(await cart_tools.view_cart_impl(session_id="session-1"))

    assert result["items"] == [{"product_id": "p1", "name": "Widget", "quantity": 2}]
    assert result["total"] == 19.98


async def test_view_cart_impl_returns_empty_cart_for_new_session(session):
    result = json.loads(await cart_tools.view_cart_impl(session_id="brand-new-session"))

    assert result["items"] == []
    assert result["total"] == 0
