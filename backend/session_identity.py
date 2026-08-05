import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import decode_access_token
from models_db import User

logger = logging.getLogger(__name__)


def _shadow_email(session_id: str) -> str:
    return f"session-{session_id}@shadow.local"


async def get_or_create_shadow_user(session: AsyncSession, session_id: str) -> User:
    """Resolves a browser session_id to a Postgres User row for cart/order
    ownership, since there is no real authentication."""
    shadow_email = _shadow_email(session_id)
    result = await session.execute(select(User).where(User.email == shadow_email))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(email=shadow_email, name="Session User")
        session.add(user)
        await session.flush()  # populate user.id without committing
        logger.info("Created shadow user %s for session %s", user.id, session_id)
    return user


async def resolve_user(session: AsyncSession, *, authorization_header: str | None, session_id: str) -> User:
    """Resolves the acting identity for a request: a real logged-in user if
    a valid JWT is present in authorization_header, otherwise the existing
    shadow-user (guest) flow keyed by session_id (#82).

    Every route/tool that currently calls get_or_create_shadow_user directly
    should call this instead: same User return shape either way, so nothing
    downstream (cart_tools.py, order_tools.py) needs to change to support
    real accounts.
    """
    if authorization_header:
        token = authorization_header.removeprefix("Bearer ").strip()
        user_id = decode_access_token(token)
        if user_id is not None:
            user = await session.get(User, user_id)
            if user is not None:
                return user
            logger.warning("Valid JWT for nonexistent user_id=%s; falling back to guest identity", user_id)

    return await get_or_create_shadow_user(session, session_id)
