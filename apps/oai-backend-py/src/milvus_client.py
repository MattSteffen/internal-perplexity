"""Milvus client pool for database operations."""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any
from urllib.parse import urlparse

from fastapi import Depends, HTTPException, Request, status
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pymilvus import MilvusClient  # type: ignore

from src.endpoints.auth import get_current_user


class MilvusSettings(BaseSettings):
    """Milvus connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="MILVUS_",
        case_sensitive=False,
    )

    uri: str = Field(
        default="http://localhost:19530",
        description="Milvus connection URI",
    )


_milvus_settings = MilvusSettings()

_DEFAULT_TTL_SECONDS = 15 * 60
_DEFAULT_MAX_SIZE = 250


def get_milvus_uri() -> str:
    """Return the configured Milvus URI (e.g. for building crawler DatabaseClientConfig)."""
    return _milvus_settings.uri


def _fingerprint_token(token: str | None) -> str | None:
    if not token:
        return None
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _PoolEntry:
    client: MilvusClient
    expires_at: float
    token_fingerprint: str | None


class MilvusClientPool:
    """Thread-safe TTL + LRU cache for per-user Milvus clients."""

    def __init__(
        self,
        uri: str,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._uri = uri
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._lock = RLock()
        self._entries: "OrderedDict[str, _PoolEntry]" = OrderedDict()

    def get(self, key: str, token: str | None) -> MilvusClient:
        now = time.monotonic()
        token_fingerprint = _fingerprint_token(token)
        with self._lock:
            self._evict_expired_locked(now)
            entry = self._entries.pop(key, None)
            if entry:
                if entry.expires_at <= now or entry.token_fingerprint != token_fingerprint:
                    self._close_client(entry.client)
                else:
                    refreshed = _PoolEntry(
                        client=entry.client,
                        expires_at=now + self._ttl_seconds,
                        token_fingerprint=entry.token_fingerprint,
                    )
                    self._entries[key] = refreshed
                    return refreshed.client

            while len(self._entries) >= self._max_size:
                _, lru_entry = self._entries.popitem(last=False)
                self._close_client(lru_entry.client)

            client = MilvusClient(uri=self._uri, token=token)
            self._entries[key] = _PoolEntry(
                client=client,
                expires_at=now + self._ttl_seconds,
                token_fingerprint=token_fingerprint,
            )
            return client

    def invalidate(self, key: str) -> None:
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry:
                self._close_client(entry.client)

    def close_all(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                self._close_client(entry.client)
            self._entries.clear()

    def _evict_expired_locked(self, now: float) -> None:
        expired_keys = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired_keys:
            entry = self._entries.pop(key, None)
            if entry:
                self._close_client(entry.client)

    @staticmethod
    def _close_client(client: MilvusClient) -> None:
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def _unwrap_user(user: dict[str, Any]) -> dict[str, Any]:
    if "user" in user and isinstance(user["user"], dict):
        return user["user"]
    return user


def derive_cache_key(user: dict[str, Any] | None, token: str | None) -> str:
    """Derive a stable cache key for the given user.

    Prefer user identifiers and only fall back to token fingerprinting when needed.
    """
    if user:
        for field in ("user_id", "id", "sub"):
            value = user.get(field)
            if isinstance(value, str) and value:
                return f"user:{value}"
        for field in ("username", "user"):
            value = user.get(field)
            if isinstance(value, str) and value:
                return f"username:{value}"
    if token:
        return f"token:{_fingerprint_token(token)}"
    raise ValueError("No stable cache key available (missing user and token)")


def get_milvus_token_for_user(user: dict[str, Any]) -> str:
    """Stub for retrieving Milvus credentials for a user.

    TODO: Replace with a real lookup (DB/Vault). Do not persist raw tokens in logs.
    """
    token = user.get("milvus_token", "")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Milvus token is required",
        )
    return token


@dataclass(frozen=True)
class MilvusClientContext:
    client: MilvusClient
    pool: MilvusClientPool
    cache_key: str
    token: str
    user: dict[str, Any]


def get_milvus_context(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> MilvusClientContext:
    user_claims = _unwrap_user(user)
    token = get_milvus_token_for_user(user_claims)
    cache_key = derive_cache_key(user_claims, token)
    pool: MilvusClientPool = request.app.state.milvus_pool
    client = pool.get(cache_key, token)
    return MilvusClientContext(
        client=client,
        pool=pool,
        cache_key=cache_key,
        token=token,
        user=user_claims,
    )


def get_milvus_client(context: MilvusClientContext = Depends(get_milvus_context)) -> MilvusClient:
    return context.client


def parse_milvus_uri(uri: str) -> tuple[str, int]:
    """Parse Milvus URI into host and port.

    Args:
        uri: e.g. http://localhost:19530 or https://milvus.example.com:19530

    Returns:
        (host, port) tuple. Port defaults to 19530 if missing.

    Raises:
        ValueError: If URI cannot be parsed.
    """
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port if parsed.port is not None else 19530
    return (host, port)


def parse_milvus_token(token: str | None) -> tuple[str, str]:
    """Parse token into username and password for DatabaseClientConfig.

    Args:
        token: Format username:password, or None for defaults.

    Returns:
        (username, password) tuple. Defaults to ("root", "Milvus") if token is None or invalid.
    """
    if not token or ":" not in token:
        return ("root", "Milvus")
    parts = token.split(":", 1)
    return (parts[0].strip() or "root", parts[1].strip() or "Milvus")
