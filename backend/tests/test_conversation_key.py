"""Tests for app/routes/chat.py's _conversation_key (#82) — keys conversation
history in Redis by user_id when a valid JWT is present, falling back to
session_id for guests, without ever hitting Postgres (see the function's own
docstring for why: a JWT's signature alone proves the user_id claim).
"""

from app.routes.chat import _conversation_key
from auth import create_access_token


class _FakeRequest:
    def __init__(self, authorization_header: str | None):
        self.headers = {"authorization": authorization_header} if authorization_header else {}


def test_returns_session_id_when_no_authorization_header():
    request = _FakeRequest(None)

    assert _conversation_key(request, "guest-session-1") == "guest-session-1"


def test_returns_prefixed_user_id_for_a_valid_bearer_token():
    # Prefixed ("user:55"), not the bare numeric id — see the function's
    # docstring: a bare "55" would collide with a guest who happens to send
    # session_id="55".
    token = create_access_token(55)
    request = _FakeRequest(f"Bearer {token}")

    assert _conversation_key(request, "irrelevant-session-id") == "user:55"


def test_returns_session_id_for_a_malformed_token():
    request = _FakeRequest("Bearer not-a-real-token")

    assert _conversation_key(request, "guest-session-2") == "guest-session-2"


def test_same_session_id_resolves_differently_authenticated_vs_guest():
    token = create_access_token(55)
    authenticated_request = _FakeRequest(f"Bearer {token}")
    guest_request = _FakeRequest(None)

    authenticated_key = _conversation_key(authenticated_request, "shared-session-id")
    guest_key = _conversation_key(guest_request, "shared-session-id")

    assert authenticated_key != guest_key
    assert authenticated_key == "user:55"
    assert guest_key == "shared-session-id"


def test_does_not_collide_when_guest_session_id_matches_a_real_user_id():
    """The exact scenario the "user:" prefix exists to prevent: a guest
    deliberately (or coincidentally) sends session_id equal to a real
    user's numeric id."""
    token = create_access_token(55)
    authenticated_request = _FakeRequest(f"Bearer {token}")
    guest_request = _FakeRequest(None)

    authenticated_key = _conversation_key(authenticated_request, "irrelevant")
    guest_key_matching_id = _conversation_key(guest_request, "55")

    assert authenticated_key != guest_key_matching_id


def test_requires_bearer_scheme_prefix():
    # A bare token with no "Bearer " scheme is not a valid Authorization
    # header (RFC 7235) — must fall back to guest, not be lenient about it.
    token = create_access_token(55)
    request = _FakeRequest(token)

    assert _conversation_key(request, "guest-session-3") == "guest-session-3"


def test_bearer_scheme_is_case_insensitive():
    token = create_access_token(55)
    request = _FakeRequest(f"bearer {token}")

    assert _conversation_key(request, "irrelevant-session-id") == "user:55"
