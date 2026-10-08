"""SQLite-backed authentication service for the local operations console."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import config


class AuthService:
    """Issue signed tokens and persist users/sessions without extra dependencies."""

    PBKDF2_ITERATIONS = 180_000

    def __init__(self) -> None:
        if len(config.auth_password) < 12 or len(config.auth_secret) < 32:
            raise RuntimeError(
                "Set AUTH_PASSWORD (12+ characters) and AUTH_SECRET (32+ characters) in .env"
            )
        self.db_path = Path("data") / "auth" / "auth.db"
        self._ensure_schema()
        self._ensure_seed_admin()

    def login(self, username: str, password: str) -> dict[str, Any]:
        user = self._get_user(username)
        if not user or not bool(user["is_active"]):
            raise ValueError("用户名或密码错误")
        if not self._verify_password(password, user["password_hash"]):
            raise ValueError("用户名或密码错误")

        now = int(time.time())
        session_id = str(uuid.uuid4())
        payload = {
            "sub": user["username"],
            "sid": session_id,
            "display_name": user["display_name"],
            "role": user["role"],
            "iat": now,
            "exp": now + max(1, config.auth_token_ttl_hours) * 3600,
        }
        token = self._sign(payload)
        self._create_session(
            session_id=session_id,
            user_id=int(user["id"]),
            token=token,
            created_at=now,
            expires_at=int(payload["exp"]),
        )
        return {
            "token": token,
            "user": self._public_user(payload),
        }

    def verify_token(self, token: str) -> dict[str, Any]:
        payload = self._verify_signature(token)
        now = int(time.time())
        if int(payload.get("exp", 0)) < now:
            raise ValueError("登录已过期")

        token_hash = self._token_hash(token)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT s.revoked_at, s.expires_at, u.username, u.display_name, u.role, u.is_active
                FROM auth_sessions s
                JOIN auth_users u ON u.id = s.user_id
                WHERE s.session_id = ? AND s.token_hash = ?
                """,
                (payload.get("sid", ""), token_hash),
            ).fetchone()
        if not row or row["revoked_at"] is not None:
            raise ValueError("登录凭证已失效")
        if int(row["expires_at"]) < now:
            raise ValueError("登录已过期")
        if not bool(row["is_active"]):
            raise ValueError("用户已禁用")

        payload.update(
            {
                "sub": row["username"],
                "display_name": row["display_name"],
                "role": row["role"],
            }
        )
        return self._public_user(payload)

    def logout(self, token: str) -> None:
        payload = self._verify_signature(token)
        with self._connect() as conn:
            conn.execute(
                "UPDATE auth_sessions SET revoked_at = ? WHERE session_id = ?",
                (int(time.time()), payload.get("sid", "")),
            )

    def change_password(self, username: str, old_password: str, new_password: str) -> None:
        if len(new_password) < 8:
            raise ValueError("新密码长度至少 8 位")
        user = self._get_user(username)
        if not user or not self._verify_password(old_password, user["password_hash"]):
            raise ValueError("旧密码不正确")
        with self._connect() as conn:
            conn.execute(
                "UPDATE auth_users SET password_hash = ?, updated_at = ? WHERE username = ?",
                (self._hash_password(new_password), int(time.time()), username),
            )
            conn.execute(
                """
                UPDATE auth_sessions
                SET revoked_at = ?
                WHERE user_id = ? AND revoked_at IS NULL
                """,
                (int(time.time()), int(user["id"])),
            )

    def list_sessions(self, username: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id, s.created_at, s.expires_at, s.revoked_at
                FROM auth_sessions s
                JOIN auth_users u ON u.id = s.user_id
                WHERE u.username = ?
                ORDER BY s.created_at DESC
                LIMIT 20
                """,
                (username,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS auth_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    password_hash TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    token_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    revoked_at INTEGER,
                    FOREIGN KEY(user_id) REFERENCES auth_users(id)
                );

                CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
                    ON auth_sessions(user_id, created_at DESC);
                """
            )

    def _ensure_seed_admin(self) -> None:
        now = int(time.time())
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM auth_users WHERE username = ?",
                (config.auth_username,),
            ).fetchone()
            if row:
                return
            conn.execute(
                """
                INSERT INTO auth_users
                    (username, display_name, role, password_hash, is_active, created_at, updated_at)
                VALUES (?, ?, 'admin', ?, 1, ?, ?)
                """,
                (
                    config.auth_username,
                    config.auth_display_name,
                    self._hash_password(config.auth_password),
                    now,
                    now,
                ),
            )

    def _get_user(self, username: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM auth_users WHERE username = ?",
                (username,),
            ).fetchone()

    def _create_session(
        self,
        session_id: str,
        user_id: int,
        token: str,
        created_at: int,
        expires_at: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO auth_sessions
                    (session_id, user_id, token_hash, created_at, expires_at, revoked_at)
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (session_id, user_id, self._token_hash(token), created_at, expires_at),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _hash_password(self, password: str) -> str:
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self.PBKDF2_ITERATIONS,
        )
        return (
            f"pbkdf2_sha256${self.PBKDF2_ITERATIONS}$"
            f"{self._b64encode(salt)}${self._b64encode(digest)}"
        )

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        try:
            scheme, iterations, salt_b64, digest_b64 = stored_hash.split("$", 3)
            if scheme != "pbkdf2_sha256":
                return False
            salt = self._b64decode(salt_b64)
            expected = self._b64decode(digest_b64)
            actual = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                int(iterations),
            )
            return hmac.compare_digest(actual, expected)
        except Exception:
            return False

    def _sign(self, payload: dict[str, Any]) -> str:
        payload_part = self._b64encode(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        return f"{payload_part}.{self._signature(payload_part)}"

    def _verify_signature(self, token: str) -> dict[str, Any]:
        try:
            payload_part, signature = token.rsplit(".", 1)
        except ValueError as exc:
            raise ValueError("无效登录凭证") from exc
        expected = self._signature(payload_part)
        if not hmac.compare_digest(signature, expected):
            raise ValueError("登录凭证签名无效")
        return json.loads(self._b64decode(payload_part).decode("utf-8"))

    def _signature(self, payload_part: str) -> str:
        digest = hmac.new(
            config.auth_secret.encode("utf-8"),
            payload_part.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return self._b64encode(digest)

    def _token_hash(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _public_user(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "username": payload.get("sub", ""),
            "display_name": payload.get("display_name", ""),
            "role": payload.get("role", "user"),
            "session_id": payload.get("sid", ""),
            "expires_at": payload.get("exp", 0),
        }

    @staticmethod
    def _b64encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    @staticmethod
    def _b64decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)


auth_service = AuthService()
