"""Structural tests for the session-based cart/checkout identity resolution
(backend/app/routes/cart.py, #55) — exercised directly against the ORM via
the same in-memory async SQLite fixture pattern as test_models_db.py, not
through HTTP/TestClient. See test_product_writes_structural.py's docstring
for why a real TestClient-based integration suite was deliberately not
attempted (a known aiosqlite/TestClient event-loop bug).
"""

import pytest_asyncio
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db import Base
from models_db import Address, Cart, User
from session_identity import get_or_create_shadow_user
from app.routes.cart import _get_or_create_cart


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as s:
        yield s

    await engine.dispose()


async def test_first_cart_action_creates_shadow_user_and_cart(session):
    user = await get_or_create_shadow_user(session, "session-1")
    cart = await _get_or_create_cart(session, user.id)
    await session.commit()

    assert user.email == "session-session-1@shadow.local"
    assert cart.user_id == user.id


async def test_repeat_cart_action_reuses_same_cart(session):
    user = await get_or_create_shadow_user(session, "session-1")
    first_cart = await _get_or_create_cart(session, user.id)
    await session.commit()

    # simulates a second /cart/add call with the same session_id
    same_user = await get_or_create_shadow_user(session, "session-1")
    second_cart = await _get_or_create_cart(session, same_user.id)
    await session.commit()

    assert first_cart.id == second_cart.id
    result = await session.execute(select(Cart))
    assert len(result.scalars().all()) == 1


async def test_checkout_address_ownership_check_rejects_other_users_address(session):
    owner = await get_or_create_shadow_user(session, "session-owner")
    other = await get_or_create_shadow_user(session, "session-other")
    address = Address(
        user_id=owner.id, street="1 Main St", city="Springfield", zip_code="00000", country="US"
    )
    session.add(address)
    await session.commit()

    # mirrors checkout()'s ownership check: Address.id == body.address_id AND Address.user_id == user.id
    result = await session.execute(
        select(Address).where(Address.id == address.id, Address.user_id == other.id)
    )
    assert result.scalar_one_or_none() is None

    result = await session.execute(
        select(Address).where(Address.id == address.id, Address.user_id == owner.id)
    )
    assert result.scalar_one_or_none() is not None
