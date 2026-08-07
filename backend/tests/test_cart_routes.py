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

from auth import create_access_token
from db import Base
from models_db import Address, Cart, User
from session_identity import get_or_create_shadow_user, resolve_user
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


async def test_same_session_id_resolves_to_different_carts_authenticated_vs_guest(session):
    """cart.py's routes now resolve identity via resolve_user (#82), not
    get_or_create_shadow_user directly — a real account's cart must stay
    fully separate from a guest cart, even if a guest request happens to
    reuse the same session_id string a logged-in user's request also sends."""
    real_user = User(email="real@example.com", name="Real User", password_hash="irrelevant-for-this-test")
    session.add(real_user)
    await session.commit()
    real_cart = await _get_or_create_cart(session, real_user.id)
    await session.commit()

    token = create_access_token(real_user.id)
    shared_session_id = "same-session-id-both-requests"

    # Authenticated request: JWT present, resolves to the real account
    authenticated_user = await resolve_user(session, authorization_header=f"Bearer {token}", session_id=shared_session_id)
    await session.commit()
    assert authenticated_user.id == real_user.id

    # Guest request: no JWT, same session_id string — must NOT resolve to the real account
    guest_user = await resolve_user(session, authorization_header=None, session_id=shared_session_id)
    await session.commit()
    assert guest_user.id != real_user.id
    assert guest_user.email == f"session-{shared_session_id}@shadow.local"

    guest_cart = await _get_or_create_cart(session, guest_user.id)
    await session.commit()
    assert guest_cart.id != real_cart.id
