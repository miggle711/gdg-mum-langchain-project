import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
