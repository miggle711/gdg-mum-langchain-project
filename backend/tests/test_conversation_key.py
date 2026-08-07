"""Tests for app/routes/chat.py's _authenticated_user_id and _conversation_key
(#82, split further in #54) — keys conversation history in Redis by user_id
when a valid JWT is present, falling back to session_id for guests, without
ever hitting Postgres (see _conversation_key's own docstring for why: a JWT's
signature alone proves the user_id claim).
"""

from app.routes.chat import _authenticated_user_id, _conversation_key
from auth import create_access_token


class _FakeRequest:
    def __init__(self, authorization_header: str | None):
        self.headers = {"authorization": authorization_header} if authorization_header else {}


def test_authenticated_user_id_returns_none_when_no_authorization_header():
    request = _FakeRequest(None)

    assert _authenticated_user_id(request) is None


def test_authenticated_user_id_returns_the_claimed_id_for_a_valid_bearer_token():
    token = create_access_token(55)
    request = _FakeRequest(f"Bearer {token}")

    assert _authenticated_user_id(request) == 55


def test_authenticated_user_id_returns_none_for_a_malformed_token():
    request = _FakeRequest("Bearer not-a-real-token")

    assert _authenticated_user_id(request) is None


def test_authenticated_user_id_requires_bearer_scheme_prefix():
    # A bare token with no "Bearer " scheme is not a valid Authorization
    # header (RFC 7235) — must return None, not be lenient about it.
    token = create_access_token(55)
    request = _FakeRequest(token)

    assert _authenticated_user_id(request) is None


def test_authenticated_user_id_bearer_scheme_is_case_insensitive():
    token = create_access_token(55)
    request = _FakeRequest(f"bearer {token}")

    assert _authenticated_user_id(request) == 55


def test_conversation_key_returns_session_id_when_user_id_is_none():
    assert _conversation_key("guest-session-1", None) == "guest-session-1"


def test_conversation_key_returns_prefixed_user_id_when_authenticated():
    # Prefixed ("user:55"), not the bare numeric id — see the function's
    # docstring: a bare "55" would collide with a guest who happens to send
    # session_id="55".
    assert _conversation_key("irrelevant-session-id", 55) == "user:55"


def test_conversation_key_does_not_collide_when_guest_session_id_matches_a_real_user_id():
    """The exact scenario the "user:" prefix exists to prevent: a guest
    deliberately (or coincidentally) sends session_id equal to a real
    user's numeric id."""
    authenticated_key = _conversation_key("irrelevant", 55)
    guest_key_matching_id = _conversation_key("55", None)

    assert authenticated_key != guest_key_matching_id


def test_conversation_key_does_not_let_a_guest_spoof_the_authenticated_prefix():
    """A malicious guest sending session_id="user:55" directly must not be
    able to produce the same key as an authenticated request for user_id=55
    (flagged by Copilot review on PR #85)."""
    authenticated_key = _conversation_key("irrelevant", 55)
    spoofing_guest_key = _conversation_key("user:55", None)

    assert authenticated_key != spoofing_guest_key
    assert spoofing_guest_key == "guest:user:55"
