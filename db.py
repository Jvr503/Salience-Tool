# db.py — SQLite database for Salience Tool
import os
import sqlite3
from contextlib import contextmanager
from cryptography.fernet import Fernet

DB_PATH = os.getenv("DB_PATH", "salience.db")

def _get_secret(name: str, default: str = "") -> str:
    try:
        import streamlit as st
        return os.getenv(name) or st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)

def _fernet():
    key = _get_secret("DB_ENCRYPTION_KEY")
    if not key:
        raise RuntimeError("DB_ENCRYPTION_KEY is not set.")
    return Fernet(key.encode() if isinstance(key, str) else key)

@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Schema ─────────────────────────────────────────────────────────────────────

def init_db():
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                email       TEXT UNIQUE NOT NULL,
                name        TEXT,
                picture     TEXT,
                role        TEXT NOT NULL DEFAULT 'user',
                manager_id  INTEGER REFERENCES users(id),
                created_at  TEXT DEFAULT (datetime('now')),
                last_login  TEXT
            );
            CREATE TABLE IF NOT EXISTS api_keys (
                service       TEXT PRIMARY KEY,
                encrypted_key TEXT NOT NULL,
                updated_at    TEXT DEFAULT (datetime('now')),
                updated_by    TEXT
            );
            CREATE TABLE IF NOT EXISTS export_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email    TEXT NOT NULL,
                filename      TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                file_data     BLOB NOT NULL,
                created_at    TEXT DEFAULT (datetime('now'))
            );
        """)

def seed_admin(email: str, name: str = "Admin"):
    """Insert admin user if they don't exist yet."""
    with _conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users (email, name, role) VALUES (?, ?, 'admin')",
            (email, name),
        )


# ── Users ──────────────────────────────────────────────────────────────────────

def upsert_user(email: str, name: str, picture: str = "") -> dict:
    """Create user on first login or refresh name/picture/last_login."""
    with _conn() as conn:
        conn.execute("""
            INSERT INTO users (email, name, picture, last_login)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(email) DO UPDATE SET
                name       = excluded.name,
                picture    = excluded.picture,
                last_login = datetime('now')
        """, (email, name, picture))
    return get_user(email)

def get_user(email: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None

def list_users() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY role, email").fetchall()
        return [dict(r) for r in rows]

def update_user_role(email: str, role: str):
    with _conn() as conn:
        conn.execute("UPDATE users SET role = ? WHERE email = ?", (role, email))

def assign_manager(user_email: str, manager_email: str):
    with _conn() as conn:
        mgr = conn.execute("SELECT id FROM users WHERE email = ?", (manager_email,)).fetchone()
        if mgr:
            conn.execute("UPDATE users SET manager_id = ? WHERE email = ?", (mgr["id"], user_email))

def remove_manager(user_email: str):
    with _conn() as conn:
        conn.execute("UPDATE users SET manager_id = NULL WHERE email = ?", (user_email,))


# ── API Keys ───────────────────────────────────────────────────────────────────

def set_api_key(service: str, plaintext: str, updated_by: str):
    encrypted = _fernet().encrypt(plaintext.encode()).decode()
    with _conn() as conn:
        conn.execute("""
            INSERT INTO api_keys (service, encrypted_key, updated_at, updated_by)
            VALUES (?, ?, datetime('now'), ?)
            ON CONFLICT(service) DO UPDATE SET
                encrypted_key = excluded.encrypted_key,
                updated_at    = datetime('now'),
                updated_by    = excluded.updated_by
        """, (service, encrypted, updated_by))

def get_api_key(service: str) -> str | None:
    with _conn() as conn:
        row = conn.execute("SELECT encrypted_key FROM api_keys WHERE service = ?", (service,)).fetchone()
    if not row:
        return None
    return _fernet().decrypt(row["encrypted_key"].encode()).decode()

def list_api_key_meta() -> list[dict]:
    """Returns service, updated_at, updated_by — never the key itself."""
    with _conn() as conn:
        rows = conn.execute("SELECT service, updated_at, updated_by FROM api_keys").fetchall()
        return [dict(r) for r in rows]


# ── Export History ─────────────────────────────────────────────────────────────

def save_export(user_email: str, filename: str, analysis_type: str, file_data: bytes):
    with _conn() as conn:
        conn.execute("""
            INSERT INTO export_history (user_email, filename, analysis_type, file_data)
            VALUES (?, ?, ?, ?)
        """, (user_email, filename, analysis_type, file_data))

def get_history(user_email: str, role: str) -> list[dict]:
    """Role-scoped history: admin=all, manager=team+self, user=self only."""
    with _conn() as conn:
        if role == "admin":
            rows = conn.execute(
                "SELECT id, user_email, filename, analysis_type, created_at "
                "FROM export_history ORDER BY created_at DESC"
            ).fetchall()
        elif role == "manager":
            mgr = conn.execute("SELECT id FROM users WHERE email = ?", (user_email,)).fetchone()
            if mgr:
                team = conn.execute(
                    "SELECT email FROM users WHERE manager_id = ? OR email = ?",
                    (mgr["id"], user_email),
                ).fetchall()
                emails = [r["email"] for r in team]
                placeholders = ",".join("?" * len(emails))
                rows = conn.execute(
                    f"SELECT id, user_email, filename, analysis_type, created_at "
                    f"FROM export_history WHERE user_email IN ({placeholders}) "
                    f"ORDER BY created_at DESC",
                    emails,
                ).fetchall()
            else:
                rows = []
        else:
            rows = conn.execute(
                "SELECT id, user_email, filename, analysis_type, created_at "
                "FROM export_history WHERE user_email = ? ORDER BY created_at DESC",
                (user_email,),
            ).fetchall()
        return [dict(r) for r in rows]

def get_export_file(export_id: int, requesting_email: str, role: str) -> tuple[str | None, bytes | None]:
    """Retrieve file bytes after permission check. Returns (filename, bytes) or (None, None)."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT user_email, filename, file_data FROM export_history WHERE id = ?",
            (export_id,),
        ).fetchone()
        if not row:
            return None, None
        if role == "admin":
            return row["filename"], bytes(row["file_data"])
        if row["user_email"] == requesting_email:
            return row["filename"], bytes(row["file_data"])
        if role == "manager":
            mgr = conn.execute("SELECT id FROM users WHERE email = ?", (requesting_email,)).fetchone()
            if mgr:
                owned = conn.execute(
                    "SELECT id FROM users WHERE email = ? AND manager_id = ?",
                    (row["user_email"], mgr["id"]),
                ).fetchone()
                if owned:
                    return row["filename"], bytes(row["file_data"])
        return None, None
