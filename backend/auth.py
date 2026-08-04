import logging
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from app.config import settings

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """Hashes a plaintext password with bcrypt (random salt per call, embedded
    in the returned hash) before it's ever persisted."""
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Checks a plaintext password against a stored bcrypt hash. bcrypt is
    one-way — this re-hashes with the salt embedded in password_hash and
    compares, it never decrypts anything."""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(user_id: int) -> str:
    """Issues a signed JWT identifying user_id, expiring after
    settings.jwt_expiry_minutes."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes)
    claims = {"sub": str(user_id), "exp": expire}
    return jwt.encode(claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> int | None:
    """Verifies a JWT's signature and expiry, returning the user_id if valid.
    Returns None (never raises) for any failure — expired, malformed, wrong
    signature, or otherwise invalid — so callers can treat "no valid token"
    as "fall back to guest identity" rather than an error."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        return None
