"""
auth.py — who is asking, from the caller's UB access token (D28).

UB calls this service with two credentials:

- `X-Serverless-Authorization`: a Google ID token for UB's runtime service
  account. Cloud Run checks it (IAM invoker) before a request gets here, and
  strips its signature.
- `Authorization: Bearer <UB access token>`: the person's own UB token, which
  UB forwards (gcp-universal-backend#166). It is checked here against UB's
  published keys, and it is the only source of the person: there is no person
  field (D28). The same token goes on to the engine for the turn, so the agent
  reads UB as the person (D2).

The app is the token's per-app audience: UB signs `aud = [appId, UB]` when an
app signed the person in, and `aud = UB` alone otherwise (app `None`).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import jwt

from .errors import ServiceError


@dataclass(frozen=True)
class Caller:
    person_id: str
    app_id: str | None
    token: str


class UbTokenVerifier:
    def __init__(self, jwks_url: str, issuer: str, audience: str) -> None:
        self._jwks = jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=300)
        self._issuer = issuer
        self._audience = audience

    def _verify(self, token: str) -> Caller:
        try:
            key = self._jwks.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                key.key,
                algorithms=["RS256"],
                issuer=self._issuer,
                audience=self._audience,
                leeway=30,
                options={"require": ["sub", "exp", "iss", "aud"]},
            )
        except jwt.PyJWKClientConnectionError as err:
            raise ServiceError(
                503, "UB_KEYS_UNAVAILABLE", "UB's signing keys could not be read."
            ) from err
        except jwt.PyJWTError as err:
            raise ServiceError(401, "INVALID_TOKEN", "The UB access token is not valid.") from err
        aud = claims["aud"]
        apps = [a for a in (aud if isinstance(aud, list) else [aud]) if a != self._audience]
        return Caller(person_id=str(claims["sub"]), app_id=apps[0] if apps else None, token=token)

    async def caller(self, authorization: str | None) -> Caller:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise ServiceError(401, "MISSING_TOKEN", "A UB access token is required.")
        # PyJWKClient fetches over urllib; keep it off the event loop.
        return await asyncio.to_thread(self._verify, token.strip())
