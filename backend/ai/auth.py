from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from dataclasses import dataclass

from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2 import id_token


class AuthenticationError(ValueError):
    pass


class AuthorizationError(PermissionError):
    pass


@dataclass(frozen=True)
class Principal:
    user_id: str
    expires_at: int
    role: str = "user"


def _secret() -> bytes:
    secret = os.getenv("AI_AUTH_SECRET", "").strip()
    if len(secret) < 32:
        raise AuthenticationError("AI_AUTH_SECRET must be configured with at least 32 characters")
    return secret.encode("utf-8")


def issue_test_token(user_id: str, ttl_seconds: int = 3600, role: str = "user") -> str:
    """Issue a signed token for local integration tests and trusted auth adapters."""
    if not user_id.strip():
        raise AuthenticationError("user_id is required")
    expires = int(time.time()) + ttl_seconds
    payload = f"{user_id}:{expires}:{role}".encode("utf-8")
    signature = hmac.new(_secret(), payload, hashlib.sha256).digest()
    enc = lambda value: base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
    return f"{enc(payload)}.{enc(signature)}"


def verify_bearer_token(token: str) -> Principal:
    try:
        raw_payload, raw_signature = token.split(".", 1)
        pad = "=" * (-len(raw_payload) % 4)
        payload = base64.urlsafe_b64decode(raw_payload + pad)
        signature = base64.urlsafe_b64decode(raw_signature + "=" * (-len(raw_signature) % 4))
        expected = hmac.new(_secret(), payload, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise AuthenticationError("Invalid bearer token")
        parts = payload.decode("utf-8").split(":")
        if len(parts) == 2:
            user_id, raw_expiry = parts; role = "user"
        elif len(parts) == 3:
            user_id, raw_expiry, role = parts
        else:
            raise AuthenticationError("Malformed bearer token")
        expires = int(raw_expiry)
        if expires <= int(time.time()):
            raise AuthenticationError("Bearer token expired")
        if not user_id:
            raise AuthenticationError("Bearer token has no user identity")
        if role not in {"user", "super_admin"}:
            raise AuthenticationError("Bearer token has invalid role")
        return Principal(user_id=user_id, expires_at=expires, role=role)
    except (ValueError, TypeError, base64.binascii.Error) as exc:
        if isinstance(exc, AuthenticationError):
            raise
        raise AuthenticationError("Malformed bearer token") from exc


def verify_google_id_token(token: str, client_id: str) -> Principal:
    if not client_id:
        raise AuthenticationError("Google sign-in is not configured")
    try:
        claims = id_token.verify_oauth2_token(token, Request(), audience=client_id)
        user_id = claims["sub"]
        expires_at = int(claims["exp"])
    except (GoogleAuthError, KeyError, TypeError, ValueError) as exc:
        raise AuthenticationError("Invalid Google ID token") from exc
    if not user_id or expires_at <= int(time.time()):
        raise AuthenticationError("Invalid Google ID token")
    return Principal(user_id=f"google:{user_id}", expires_at=expires_at)
