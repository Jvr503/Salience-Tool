# pages/1_Admin.py — Admin panel for the Salience Tool
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import auth
import db

st.set_page_config(page_title="Admin — Salience Tool", page_icon="⚙️", layout="wide")

# ── Auth gate ──────────────────────────────────────────────────────────────────
user = auth.require_auth()

if not auth.is_admin():
    st.error("⛔ Admin access required.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("propellic-logo-png.png", width=150)
    except Exception:
        pass
    st.markdown(f"**{user['name']}**")
    st.caption(user["email"])
    st.caption("Role: Admin")
    if st.button("Sign out"):
        auth.logout()

# ── Page ───────────────────────────────────────────────────────────────────────
st.title("⚙️ Admin Panel")

tab_users, tab_keys, tab_history = st.tabs(["👥 Users", "🔑 API Keys", "📂 Export History"])


# ══════════════════════════════════════════════════════════════════════════════
# USERS
# ══════════════════════════════════════════════════════════════════════════════

with tab_users:
    st.subheader("All Users")
    users = db.list_users()

    if users:
        display_cols = ["email", "name", "role", "last_login"]
        df = pd.DataFrame(users)[display_cols]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No users yet — they appear here after their first sign-in.")

    st.divider()

    col_role, col_mgr = st.columns(2)

    with col_role:
        st.subheader("Change Role")
        if users:
            emails = [u["email"] for u in users]
            target_email = st.selectbox("User", emails, key="role_target")
            current_role = next((u["role"] for u in users if u["email"] == target_email), "user")
            new_role = st.selectbox(
                "New role",
                ["user", "manager", "admin"],
                index=["user", "manager", "admin"].index(current_role),
                key="new_role_select",
            )
            if st.button("Update Role"):
                if target_email == user["email"] and new_role != "admin":
                    st.error("You can't demote yourself.")
                else:
                    db.update_user_role(target_email, new_role)
                    st.success(f"Updated **{target_email}** → **{new_role}**")
                    st.rerun()

    with col_mgr:
        st.subheader("Assign Manager")
        if users:
            non_admin = [u["email"] for u in users if u["role"] != "admin"]
            managers  = [u["email"] for u in users if u["role"] in ("manager", "admin")]

            if non_admin and managers:
                assign_target = st.selectbox("User to assign", non_admin, key="assign_target")
                assign_mgr    = st.selectbox("Assign to manager", managers, key="assign_manager")
                col_a, col_b  = st.columns(2)
                with col_a:
                    if st.button("Assign"):
                        db.assign_manager(assign_target, assign_mgr)
                        st.success(f"**{assign_target}** → manager: **{assign_mgr}**")
                        st.rerun()
                with col_b:
                    if st.button("Remove manager"):
                        db.remove_manager(assign_target)
                        st.success(f"Removed manager from **{assign_target}**")
                        st.rerun()
            else:
                st.info("Need at least one manager/admin and one non-admin user.")


# ══════════════════════════════════════════════════════════════════════════════
# API KEYS
# ══════════════════════════════════════════════════════════════════════════════

with tab_keys:
    st.subheader("API Key Management")
    st.info(
        "Keys are **encrypted at rest** using Fernet symmetric encryption. "
        "Once saved they cannot be viewed — only replaced."
    )

    meta = {m["service"]: m for m in db.list_api_key_meta()}

    services = [
        ("anthropic",  "Anthropic (Claude API)"),
        ("google_nlp", "Google Cloud NLP"),
    ]

    for service, label in services:
        with st.expander(f"🔑 {label}", expanded=True):
            if service in meta:
                m = meta[service]
                st.success(f"✓ Key is set — last updated **{m['updated_at']}** by {m['updated_by']}")
            else:
                st.warning("Not set — the tool will fall back to the .env value if present.")

            new_key = st.text_input(
                f"Paste new {label} key",
                type="password",
                key=f"input_key_{service}",
                placeholder="Paste key here to update…",
            )
            if st.button(f"Save {label} key", key=f"save_{service}"):
                if new_key.strip():
                    db.set_api_key(service, new_key.strip(), user["email"])
                    st.success("Saved and encrypted.")
                    st.rerun()
                else:
                    st.error("Key cannot be empty.")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT HISTORY
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.subheader("All Export History")

    history = db.get_history(user["email"], "admin")

    if not history:
        st.info("No exports yet.")
    else:
        df = pd.DataFrame(history)[["id", "user_email", "filename", "analysis_type", "created_at"]]
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**Download an export by ID:**")
        export_id = st.number_input("Export ID", min_value=1, step=1, key="admin_dl_id")
        if st.button("Fetch for download"):
            fname, fdata = db.get_export_file(int(export_id), user["email"], "admin")
            if fdata:
                st.download_button(
                    f"⬇ Download {fname}",
                    data=fdata,
                    file_name=fname,
                    key="admin_dl_btn",
                )
            else:
                st.error("Export not found.")
