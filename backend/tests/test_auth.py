"""Tests for auth.py's password hashing and JWT helpers (#82)."""

from datetime import datetime, timedelta, timezone

from jose import jwt

from app.config import settings
from auth import create_access_token, decode_access_token, hash_password, verify_password


def test_hash_password_produces_a_bcrypt_hash_distinct_from_the_input():
    hashed = hash_password("correct horse battery staple")

    assert hashed != "correct horse battery staple"
    assert hashed.startswith("$2b$")


def test_hash_password_uses_a_random_salt_per_call():
    first = hash_password("same password")
    second = hash_password("same password")

    assert first != second


def test_verify_password_accepts_the_correct_password():
    hashed = hash_password("correct horse battery staple")

    assert verify_password("correct horse battery staple", hashed) is True


def test_verify_password_rejects_an_incorrect_password():
    hashed = hash_password("correct horse battery staple")

    assert verify_password("wrong password", hashed) is False


def test_create_access_token_round_trips_through_decode_access_token():
    token = create_access_token(42)

    assert decode_access_token(token) == 42


def test_decode_access_token_returns_none_for_malformed_token():
    assert decode_access_token("not.a.valid.token") is None


def test_decode_access_token_returns_none_for_empty_token():
    assert decode_access_token("") is None


def test_decode_access_token_returns_none_for_wrong_signature():
    token = jwt.encode({"sub": "42", "exp": datetime.now(timezone.utc) + timedelta(minutes=5)}, "a-different-secret", algorithm=settings.jwt_algorithm)

    assert decode_access_token(token) is None


def test_decode_access_token_returns_none_for_expired_token():
    expired_claims = {"sub": "42", "exp": datetime.now(timezone.utc) - timedelta(minutes=5)}
    expired_token = jwt.encode(expired_claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    assert decode_access_token(expired_token) is None


def test_decode_access_token_returns_none_for_missing_sub_claim():
    claims_without_sub = {"exp": datetime.now(timezone.utc) + timedelta(minutes=5)}
    token = jwt.encode(claims_without_sub, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    assert decode_access_token(token) is None
