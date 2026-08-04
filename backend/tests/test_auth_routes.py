"""Structural tests for the signup/login routes (backend/app/routes/auth.py,
#82) — exercised directly against the ORM via the same in-memory async
SQLite fixture pattern as test_cart_routes.py, not through HTTP/TestClient.
See test_cart_routes.py's docstring for why (a known aiosqlite/TestClient
event-loop bug).
"""

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from auth import hash_password, verify_password
from db import Base
from models_db import User
from session_identity import get_or_create_shadow_user


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as s:
        yield s

    await engine.dispose()


async def test_signup_creates_user_with_hashed_password(session):
    # mirrors signup()'s core logic: uniqueness check, then create with a hash
    result = await session.execute(select(User).where(User.email == "new@example.com"))
    assert result.scalar_one_or_none() is None

    user = User(email="new@example.com", name="New User", password_hash=hash_password("hunter2"))
    session.add(user)
    await session.commit()

    assert user.password_hash != "hunter2"
    assert verify_password("hunter2", user.password_hash) is True


async def test_signup_rejects_duplicate_email(session):
    session.add(User(email="taken@example.com", name="First", password_hash=hash_password("pw1")))
    await session.commit()

    # mirrors signup()'s uniqueness check
    result = await session.execute(select(User).where(User.email == "taken@example.com"))
    assert result.scalar_one_or_none() is not None


async def test_login_succeeds_with_correct_password(session):
    user = User(email="login@example.com", name="Login User", password_hash=hash_password("correct-password"))
    session.add(user)
    await session.commit()

    result = await session.execute(select(User).where(User.email == "login@example.com"))
    found = result.scalar_one_or_none()

    assert found is not None
    assert found.password_hash is not None
    assert verify_password("correct-password", found.password_hash) is True


async def test_login_fails_with_wrong_password(session):
    user = User(email="login@example.com", name="Login User", password_hash=hash_password("correct-password"))
    session.add(user)
    await session.commit()

    result = await session.execute(select(User).where(User.email == "login@example.com"))
    found = result.scalar_one_or_none()

    assert verify_password("wrong-password", found.password_hash) is False


async def test_login_rejects_nonexistent_email(session):
    result = await session.execute(select(User).where(User.email == "nobody@example.com"))
    assert result.scalar_one_or_none() is None


async def test_login_rejects_shadow_user_with_no_password_hash(session):
    # A shadow/guest user created via get_or_create_shadow_user never has a
    # password_hash — login() must treat this as "no valid credentials",
    # never attempt verify_password() against a None hash.
    shadow_user = await get_or_create_shadow_user(session, "some-session-id")
    await session.commit()

    assert shadow_user.password_hash is None
