import logging
import sys
import os

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from auth import create_access_token, hash_password, verify_password
from db import get_session
from models_db import User
from app.models import AuthResponse, LoginRequest, SignupRequest

router = APIRouter()


@router.post("/auth/signup", status_code=201)
async def signup(body: SignupRequest) -> AuthResponse:
    try:
        async with get_session() as session:
            result = await session.execute(select(User).where(User.email == body.email))
            if result.scalar_one_or_none() is not None:
                raise HTTPException(status_code=409, detail="An account with this email already exists")

            user = User(
                email=body.email,
                name=body.name,
                password_hash=hash_password(body.password),
            )
            session.add(user)
            await session.commit()

        return AuthResponse(access_token=create_access_token(user.id), user_id=user.id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Exception in signup: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/login")
async def login(body: LoginRequest) -> AuthResponse:
    try:
        async with get_session() as session:
            result = await session.execute(select(User).where(User.email == body.email))
            user = result.scalar_one_or_none()

            # Same "invalid email or password" message for both a missing
            # account and a wrong password — don't reveal whether an email
            # is registered.
            if user is None or user.password_hash is None or not verify_password(body.password, user.password_hash):
                raise HTTPException(status_code=401, detail="Invalid email or password")

        return AuthResponse(access_token=create_access_token(user.id), user_id=user.id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Exception in login: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
