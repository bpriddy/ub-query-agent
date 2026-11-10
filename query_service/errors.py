"""errors.py — the service's errors: an HTTP status, a stable code, a message.
The body is `{"code", "message"}`, the shape UB's own errors use."""

from __future__ import annotations


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def body(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}
