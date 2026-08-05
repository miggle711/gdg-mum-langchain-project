"""Tests for session_identity.py's resolve_user (#82) — the shared identity
resolver used by every route/tool that currently calls
get_or_create_shadow_user. Uses the same in-memory async SQLite fixture
pattern as test_session_identity.py.
"""

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from auth import create_access_token
from db import Base
from session_identity import get_or_create_shadow_user, resolve_user


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as s:
        yield s

    await engine.dispose()


async def test_resolves_to_the_real_user_for_a_valid_token(session):
    real_user = await get_or_create_shadow_user(session, "not-used-for-this-user")
    real_user.name = "Real Account Holder"
    await session.commit()
    token = create_access_token(real_user.id)

    resolved = await resolve_user(session, authorization_header=f"Bearer {token}", session_id="irrelevant")

    assert resolved.id == real_user.id
    assert resolved.name == "Real Account Holder"


async def test_falls_back_to_guest_when_no_authorization_header(session):
    resolved = await resolve_user(session, authorization_header=None, session_id="guest-session-1")
    await session.commit()

    assert resolved.email == "session-guest-session-1@shadow.local"


async def test_falls_back_to_guest_for_a_malformed_token(session):
    resolved = await resolve_user(session, authorization_header="Bearer not-a-real-token", session_id="guest-session-2")
    await session.commit()

    assert resolved.email == "session-guest-session-2@shadow.local"


async def test_falls_back_to_guest_when_token_points_to_a_nonexistent_user(session):
    token = create_access_token(999999)

    resolved = await resolve_user(session, authorization_header=f"Bearer {token}", session_id="guest-session-3")
    await session.commit()

    assert resolved.email == "session-guest-session-3@shadow.local"


async def test_strips_bearer_prefix_correctly(session):
    real_user = await get_or_create_shadow_user(session, "not-used")
    await session.commit()
    token = create_access_token(real_user.id)

    # No "Bearer " prefix at all should also resolve correctly, since the
    # resolver strips the prefix rather than requiring it.
    resolved = await resolve_user(session, authorization_header=token, session_id="irrelevant")

    assert resolved.id == real_user.id
