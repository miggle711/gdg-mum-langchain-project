"""Tests for session_identity.py's shadow-user resolution (#55) — uses the
same in-memory async SQLite fixture pattern as test_models_db.py.
"""

import pytest_asyncio
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db import Base
from models_db import User
from session_identity import get_or_create_shadow_user


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


async def test_creates_shadow_user_with_derived_email(session):
    user = await get_or_create_shadow_user(session, "abc-123")
    await session.commit()

    assert user.email == "session-abc-123@shadow.local"
    assert user.name == "Session User"
    assert user.id is not None


async def test_same_session_id_returns_same_user(session):
    first = await get_or_create_shadow_user(session, "abc-123")
    await session.commit()

    second = await get_or_create_shadow_user(session, "abc-123")
    await session.commit()

    assert first.id == second.id
    result = await session.execute(select(User))
    assert len(result.scalars().all()) == 1


async def test_different_session_ids_get_distinct_users(session):
    first = await get_or_create_shadow_user(session, "session-a")
    second = await get_or_create_shadow_user(session, "session-b")
    await session.commit()

    assert first.id != second.id
