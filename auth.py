# auth.py — Google OAuth for Salience Tool
import os
import streamlit as st
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import db

ALLOWED_DOMAIN = "propellic.com"

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


def _creds():
    secrets = getattr(st, "secrets", {})
    return (
        os.getenv("GOOGLE_CLIENT_ID") or secrets.get("GOOGLE_CLIENT_ID", ""),
        os.getenv("GOOGLE_CLIENT_SECRET") or secrets.get("GOOGLE_CLIENT_SECRET", ""),
        os.getenv("REDIRECT_URI") or secrets.get("REDIRECT_URI", "http://localhost:8501/"),
    )


def _flow() -> Flow:
    client_id, client_secret, redirect_uri = _creds()
    return Flow.from_client_config(
        {
            "web": {
                "client_id":     client_id,
                "client_secret": client_secret,
                "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
                "token_uri":     "https://oauth2.googleapis.com/token",
                "redirect_uris": [redirect_uri],
            }
        },
        scopes=SCOPES,
        redirect_uri=redirect_uri,
        autogenerate_code_verifier=False,
    )


def get_auth_url() -> str:
    flow = _flow()
    auth_url, state = flow.authorization_url(prompt="select_account", hd="propellic.com")
    st.session_state["_oauth_state"] = state
    st.session_state["_oauth_flow"] = flow  # preserve code_verifier for callback
    return auth_url


def _handle_callback() -> bool:
    """Detect OAuth callback in query params and exchange code for user info."""
    if st.session_state.get("user"):
        return True

    code = st.query_params.get("code")
    if not code:
        return False

    try:
        # Allow HTTP for local dev; production should use HTTPS
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        flow = st.session_state.pop("_oauth_flow", None) or _flow()
        flow.fetch_token(code=code)

        client_id, _, _ = _creds()
        id_info = id_token.verify_oauth2_token(
            flow.credentials.id_token,
            google_requests.Request(),
            client_id,
            clock_skew_in_seconds=10,
        )

        email   = id_info.get("email", "")
        name    = id_info.get("name", "User")
        picture = id_info.get("picture", "")

        if not email.endswith(f"@{ALLOWED_DOMAIN}"):
            st.error(f"Access is restricted to @{ALLOWED_DOMAIN} accounts.")
            st.query_params.clear()
            return False

        user = db.upsert_user(email, name, picture)
        st.session_state["user"] = user
        st.query_params.clear()
        st.rerun()

    except Exception as exc:
        st.error(f"Sign-in failed: {exc}")
        st.query_params.clear()

    return False


def require_auth() -> dict:
    """
    Call at the top of every page.
    Returns the logged-in user dict, or stops execution and shows the login screen.
    """
    if _handle_callback():
        return st.session_state["user"]
    _show_login()
    st.stop()


def logout():
    st.session_state.pop("user", None)
    st.rerun()


def current_user() -> dict | None:
    return st.session_state.get("user")


def is_admin() -> bool:
    u = current_user()
    return bool(u and u.get("role") == "admin")


def is_manager_or_above() -> bool:
    u = current_user()
    return bool(u and u.get("role") in ("admin", "manager"))


# ── Login page ─────────────────────────────────────────────────────────────────

def _show_login():
    st.markdown("""
        <style>
        section[data-testid="stSidebar"] { display: none; }
        .login-wrap { text-align: center; padding: 4rem 1rem; }
        .google-btn {
            display: inline-flex; align-items: center; gap: 12px;
            background: #fff; color: #333; border: 1px solid #ddd;
            border-radius: 8px; padding: 12px 28px;
            font-size: 16px; font-weight: 500; text-decoration: none;
            box-shadow: 0 2px 4px rgba(0,0,0,.12); margin-top: 1.5rem;
        }
        .google-btn:hover { box-shadow: 0 4px 10px rgba(0,0,0,.18); }
        </style>
    """, unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        try:
            st.image("propellic-logo-png.png", width=200)
        except Exception:
            pass
        st.markdown("## Salience Analyzer")
        st.markdown("Sign in with your **Propellic Google account** to continue.")

        client_id, client_secret, _ = _creds()
        if not client_id or not client_secret:
            st.warning(
                "Google OAuth is not configured. "
                "Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to your .env file. "
                "See README for setup instructions."
            )
            return

        auth_url = get_auth_url()
        st.markdown(
            f'<button onclick="window.top.location.href=\'{auth_url}\'" class="google-btn">'
            f'<img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" width="20">'
            f'Sign in with Google</button>',
            unsafe_allow_html=True,
        )
