# Salience Tool — Propellic
# Created by Javier Hernandez

import base64
import json
import io
import os
import re
import requests
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.cloud import language_v1
import anthropic

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")

# ── Helpers ───────────────────────────────────────────────────────────────────

def first_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return parts[0].strip()


def fetch_page_elements(url: str) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome Safari"
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    meta_title = soup.title.get_text(strip=True) if soup.title else ""

    meta_desc = ""
    md = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if md and md.get("content"):
        meta_desc = md.get("content", "").strip()

    h1_text = ""
    h1 = soup.find("h1")
    if h1:
        h1_text = h1.get_text(" ", strip=True)

    after_h1_text = ""
    if h1:
        p = h1.find_next("p")
        if p:
            after_h1_text = p.get_text(" ", strip=True)

    fs = first_sentence(after_h1_text)

    return {
        "meta_title": meta_title,
        "meta_description": meta_desc,
        "h1": h1_text,
        "first_sentence_after_h1": fs,
        "prefill_original": "\n\n".join([t for t in [h1_text, fs] if t]),
    }


def analyze_text_salience(text: str) -> dict:
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_entities(document=document)
    result = {}
    for entity in response.entities:
        result[entity.name] = {
            "Type": language_v1.Entity.Type(entity.type_).name,
            "Salience": float(entity.salience),
        }
    return result


def top_entity(entities: dict):
    if not entities:
        return None, None
    top = max(entities.items(), key=lambda x: x[1]["Salience"])
    return top[0], top[1]["Salience"]


def wc(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def score_color(val: float) -> str:
    if val >= 0.60: return "#16a34a"
    if val >= 0.40: return "#ca8a04"
    if val >= 0.20: return "#ea580c"
    return "#dc2626"


def style_score(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if val >= 0.60: return "color: #16a34a; font-weight: 600"
    if val >= 0.40: return "color: #ca8a04; font-weight: 600"
    if val >= 0.20: return "color: #ea580c; font-weight: 600"
    return "color: #dc2626; font-weight: 600"


def copy_button(text: str, key: str = "") -> None:
    """Renders a clipboard copy button via JS component."""
    if not text.strip():
        return
    # base64-encode the text so it's safe to embed directly in an HTML attribute
    # (avoids quote-escaping issues that break onclick when text contains " or ')
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    components.html(
        f"""<button
          onclick="var t=atob('{b64}');navigator.clipboard.writeText(t).then(()=>{{
            this.textContent='&#10003; Copied!';
            setTimeout(()=>this.textContent='&#128203; Copy',2000);
          }})"
          style="background:#E21A6B;color:white;border:none;border-radius:6px;
                 padding:5px 14px;cursor:pointer;font-size:12px;font-weight:bold;
                 font-family:Montserrat,sans-serif;">
          &#128203; Copy
        </button>""",
        height=38,
    )


# ── Credentials ───────────────────────────────────────────────────────────────

cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if not cred:
    st.error("GOOGLE_APPLICATION_CREDENTIALS is not set. Add it to ~/.zprofile or .env, then restart.")
    st.stop()

cred_path = os.path.expanduser(cred)
if not os.path.exists(cred_path):
    st.error(f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {cred}")
    st.stop()

# ── Claude helpers ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_claude_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


_SHORT_ELEMENTS = {"H1", "Meta title"}


def generate_optimized_text(original: str, target_entity: str, source_label: str = "Pasted text") -> str:
    client = get_claude_client(ANTHROPIC_API_KEY)

    if source_label in _SHORT_ELEMENTS:
        word_cnt = len(original.split())
        prompt = f"""You are optimizing a {source_label} for Google Cloud Natural Language API entity salience.

Goal: Rewrite this {source_label} so that "{target_entity}" is the #1 entity by salience.

Rules:
1) This is a {source_label} — keep it as a short heading. Do NOT write a sentence or paragraph.
2) Keep the output at roughly {word_cnt} words (±2 words maximum). Never expand it.
3) Place "{target_entity}" as the first or second word of the heading.
4) Remove or demote any competing named entities — replace them with generic descriptors if needed.
5) Return ONLY the rewritten {source_label}. No punctuation at the end unless the original had it. No explanation.

Original {source_label}:
{original}""".strip()
    else:
        prompt = f"""You are optimizing text for Google Cloud Natural Language API entity salience.

Goal: The entity "{target_entity}" must be the #1 entity by salience when analyzed with Google NLP.

Rewrite rules:
1) Put "{target_entity}" in the first sentence, within the first 8–10 words.
2) Make "{target_entity}" the grammatical subject of most sentences (active voice).
3) Use the exact string "{target_entity}" multiple times naturally.
4) Demote competing entities — rewrite them as generic references after the first mention.
5) Do not introduce new named entities not in the original text.
6) Same length or shorter — do NOT expand the original text.

Return ONLY the rewritten text. No explanation. No **.

Original text:
{original}""".strip()

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def suggest_entity_for_keyword(entities: list, keyword: str) -> str:
    client = get_claude_client(ANTHROPIC_API_KEY)
    entity_list = ", ".join(f'"{e}"' for e in entities[:20])
    prompt = (
        f'You are an SEO specialist. A page contains these named entities: {entity_list}\n\n'
        f'A user wants to optimize this page to rank for the keyword: "{keyword}"\n\n'
        f'Which SINGLE entity from the list should be made the most salient (dominant subject) '
        f'to best align this page with that keyword from an SEO perspective? '
        f'Choose the entity that most directly represents the searcher\'s intent behind "{keyword}" '
        f'— not necessarily the brand or the most prominent entity currently on the page.\n\n'
        f'Reply with ONLY the entity name, exactly as written in the list above. No explanation.'
    )
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip().strip('"')


# ── Branding / CSS ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    .stButton>button {
        background-color: #E21A6B; color: white; font-weight: bold;
        border: none; border-radius: 8px; padding: 0.6em 1.4em;
    }
    .stButton>button:hover { background-color: #c0175d; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.image("propellic-logo-png.png", width=180)
st.markdown(
    """
    <div style="text-transform:uppercase;color:#E21A6B;font-weight:bold;font-size:14px;margin-bottom:0.5rem;">
      Salience Analyzer
    </div>
    <h1 style='color:white;margin-top:0;'>Text Analysis with Google NLP</h1>
    """,
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────

defaults: dict = {
    # URL tab
    "page_url": "",
    "url_meta_title": "",
    "url_meta_description": "",
    "url_h1": "",
    "url_first_sentence": "",
    "url_element_scores": {},   # {key: {"entity": str, "score": float}}
    # Paste tab
    "original_text": "",
    # Variations
    "variation_text_1": "",
    "variation_text_2": "",
    "target_entity_1": "",
    "target_entity_2": "",
    "variation_source_1": "Pasted text",
    "variation_source_2": "Pasted text",
    "assign_to": "Variation 1",
    "entity_picker_1": "",
    "entity_picker_2": "",
    # Results
    "display_df": None,
    "claude_error": "",
    "analyze_v1_label": "",
    "analyze_v2_label": "",
    "selected_original_key": "",
    # Keyword suggestion
    "target_keyword": "",
    "keyword_suggestion": "",
    # History
    "session_history": [],
    # Bulk
    "bulk_urls_input": "",
    "bulk_target_entity": "",
    "bulk_results_df": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.get("assign_to") not in ["Variation 1", "Variation 2"]:
    st.session_state["assign_to"] = "Variation 1"

# ── Source options ─────────────────────────────────────────────────────────────

_SOURCE_OPTIONS = ["Pasted text", "H1", "First sentence after H1", "Meta title", "Meta description"]
_SOURCE_KEY_MAP = {
    "Pasted text": "original_text",
    "H1": "url_h1",
    "First sentence after H1": "url_first_sentence",
    "Meta title": "url_meta_title",
    "Meta description": "url_meta_description",
}

# ── Callbacks ──────────────────────────────────────────────────────────────────

def clear_all():
    skip = {"session_history"}
    for k, v in defaults.items():
        if k not in skip:
            st.session_state[k] = v if not isinstance(v, (dict, list)) else type(v)()


def use_as_original(key: str):
    st.session_state["original_text"] = st.session_state.get(key, "")
    st.session_state["selected_original_key"] = key


def assign_selected_entity():
    df = st.session_state.get("display_df")
    table_state = st.session_state.get("entity_table")
    if df is None or table_state is None:
        return
    rows = table_state.selection.rows
    if not rows:
        return
    selected_entity = df.iloc[rows[0]]["Entity"]
    if st.session_state.get("assign_to") == "Variation 1":
        st.session_state["target_entity_1"] = selected_entity
    else:
        st.session_state["target_entity_2"] = selected_entity


def pick_entity(variation_num: int):
    key = f"entity_picker_{variation_num}"
    val = st.session_state.get(key, "")
    if val:
        st.session_state[f"target_entity_{variation_num}"] = val
        st.session_state[key] = ""   # reset picker to placeholder


def on_generate_with_claude():
    if not ANTHROPIC_API_KEY:
        st.session_state["claude_error"] = "ANTHROPIC_API_KEY is not set. Add it to .env and restart."
        return

    is_v1 = st.session_state["assign_to"] == "Variation 1"
    target = st.session_state["target_entity_1"] if is_v1 else st.session_state["target_entity_2"]
    source_label = st.session_state["variation_source_1"] if is_v1 else st.session_state["variation_source_2"]
    source_key = _SOURCE_KEY_MAP[source_label]
    original = st.session_state.get(source_key, "")

    if not original.strip():
        st.session_state["claude_error"] = (
            f'No text found for "{source_label}". '
            + ("Paste text in the Paste tab first." if source_key == "original_text" else "Load a URL first.")
        )
        return
    if not target.strip():
        st.session_state["claude_error"] = "Enter a target entity first."
        return

    st.session_state["claude_error"] = ""
    try:
        new_text = generate_optimized_text(original, target, source_label)
    except Exception as e:
        st.session_state["claude_error"] = f"Claude request failed: {e}"
        return

    if is_v1:
        st.session_state["variation_text_1"] = new_text
    else:
        st.session_state["variation_text_2"] = new_text

    st.toast("Text generated successfully.")


# ── Score guide markdown ───────────────────────────────────────────────────────

SCORE_GUIDE = """
<div style="background:rgba(226,26,107,0.08);border-left:4px solid #E21A6B;
            padding:10px 14px;border-radius:4px;font-size:13px;margin:12px 0;">
  <strong>Salience Score Guide</strong> &nbsp;—&nbsp;
  <span style="color:#16a34a;font-weight:600">0.60+</span>&nbsp;Excellent: entity clearly dominates &nbsp;
  <span style="color:#ca8a04;font-weight:600">0.40–0.59</span>&nbsp;Good: strong signal &nbsp;
  <span style="color:#ea580c;font-weight:600">0.20–0.39</span>&nbsp;Moderate: competing entities dilute focus &nbsp;
  <span style="color:#dc2626;font-weight:600">&lt; 0.20</span>&nbsp;Weak: entity is not the clear subject
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_single, tab_bulk = st.tabs(["Single Analysis", "Bulk Analysis"])

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab_single:

    # ── Input tabs ────────────────────────────────────────────────────────────
    input_tab_paste, input_tab_url = st.tabs(["Paste text", "Load from URL"])

    with input_tab_url:
        st.text_input("Page URL", key="page_url")

        if st.button("Load page elements", key="load_page_elements"):
            if not st.session_state["page_url"].strip():
                st.error("Paste a URL first.")
            else:
                with st.spinner("Fetching page and scoring elements…"):
                    try:
                        data = fetch_page_elements(st.session_state["page_url"].strip())
                        st.session_state["url_meta_title"] = data.get("meta_title", "")
                        st.session_state["url_meta_description"] = data.get("meta_description", "")
                        st.session_state["url_h1"] = data.get("h1", "")
                        st.session_state["url_first_sentence"] = data.get("first_sentence_after_h1", "")
                        st.session_state["original_text"] = data.get("prefill_original", "")

                        # Auto-score each element
                        elem_map = {
                            "url_meta_title": st.session_state["url_meta_title"],
                            "url_meta_description": st.session_state["url_meta_description"],
                            "url_h1": st.session_state["url_h1"],
                            "url_first_sentence": st.session_state["url_first_sentence"],
                        }
                        scores = {}
                        for key, text_val in elem_map.items():
                            if text_val.strip():
                                try:
                                    ents = analyze_text_salience(text_val)
                                    te, ts = top_entity(ents)
                                    scores[key] = {"entity": te, "score": ts}
                                except Exception:
                                    scores[key] = {"entity": None, "score": None}
                        st.session_state["url_element_scores"] = scores
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to fetch URL: {e}")

        # Display elements with inline scores + Use as original buttons
        elem_display = [
            ("Meta title", "url_meta_title", False),
            ("Meta description", "url_meta_description", True),
            ("H1", "url_h1", False),
            ("First sentence after H1", "url_first_sentence", True),
        ]
        elem_scores = st.session_state.get("url_element_scores", {})

        for label, key, multiline in elem_display:
            text_val = st.session_state.get(key, "")
            score_info = elem_scores.get(key, {})
            col_field, col_score, col_btn = st.columns([5, 3, 2])

            with col_field:
                if multiline:
                    st.text_area(label, value=text_val, height=80, disabled=True)
                else:
                    st.text_input(label, value=text_val, disabled=True)

            with col_score:
                if score_info.get("entity"):
                    te = score_info["entity"]
                    ts = score_info["score"]
                    color = score_color(ts)
                    st.markdown(
                        f'<div style="padding-top:{"26" if not multiline else "6"}px;font-size:12px;color:#9ca3af;">'
                        f'Top entity:<br>'
                        f'<span style="color:{color};font-weight:600">{te}</span> '
                        f'<span style="color:{color};font-weight:600">({ts:.2f})</span></div>',
                        unsafe_allow_html=True,
                    )

            with col_btn:
                if text_val.strip():
                    is_selected = st.session_state.get("selected_original_key") == key
                    st.markdown(f'<div style="padding-top:{"24" if not multiline else "4"}px;">', unsafe_allow_html=True)
                    st.button("Analyze this element", key=f"use_{key}",
                              on_click=use_as_original, args=(key,))
                    if is_selected:
                        st.markdown(
                            '<div style="font-size:11px;color:#16a34a;font-weight:700;'
                            'background:rgba(22,163,74,0.12);border:1px solid #16a34a;'
                            'border-radius:4px;padding:2px 7px;margin-top:4px;text-align:center;">'
                            '✓ Active original</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown('</div>', unsafe_allow_html=True)

    with input_tab_paste:
        st.text_area(
            "Paste the original content you want to analyze:",
            height=100,
            key="original_text",
        )
        orig_wc = wc(st.session_state.get("original_text", ""))
        if orig_wc:
            st.caption(f"{orig_wc} words")

    # ── Variation containers ───────────────────────────────────────────────────
    st.markdown("---")
    orig_wc_for_delta = wc(st.session_state.get("original_text", ""))

    with st.container(border=True):
        st.markdown("#### Variation 1")
        st.text_input("Target entity for Variation 1", key="target_entity_1", placeholder="Type any entity name…")
        _df_for_picker = st.session_state.get("display_df")
        if _df_for_picker is not None and not _df_for_picker.empty:
            st.selectbox(
                "Or pick from last analysis:",
                options=[""] + _df_for_picker["Entity"].tolist(),
                format_func=lambda x: "— pick an entity —" if x == "" else x,
                key="entity_picker_1",
                on_change=pick_entity, args=(1,),
                label_visibility="collapsed",
            )
        st.selectbox(
            "Element to optimize", options=_SOURCE_OPTIONS, key="variation_source_1",
            help="URL elements are available after loading a URL above.",
        )
        st.text_area(
            "Variation 1 text (paste manually or click Generate below):",
            height=100, key="variation_text_1",
        )
        v1_wc = wc(st.session_state.get("variation_text_1", ""))
        if v1_wc:
            delta = v1_wc - orig_wc_for_delta if orig_wc_for_delta else 0
            delta_str = f" ({'+' if delta >= 0 else ''}{delta} vs original)" if orig_wc_for_delta else ""
            st.caption(f"{v1_wc} words{delta_str}")
        copy_button(st.session_state.get("variation_text_1", ""), key="v1")

    with st.container(border=True):
        st.markdown("#### Variation 2")
        st.text_input("Target entity for Variation 2", key="target_entity_2", placeholder="Type any entity name…")
        if _df_for_picker is not None and not _df_for_picker.empty:
            st.selectbox(
                "Or pick from last analysis:",
                options=[""] + _df_for_picker["Entity"].tolist(),
                format_func=lambda x: "— pick an entity —" if x == "" else x,
                key="entity_picker_2",
                on_change=pick_entity, args=(2,),
                label_visibility="collapsed",
            )
        st.selectbox(
            "Element to optimize", options=_SOURCE_OPTIONS, key="variation_source_2",
            help="URL elements are available after loading a URL above.",
        )
        st.text_area(
            "Variation 2 text (paste manually or click Generate below):",
            height=100, key="variation_text_2",
        )
        v2_wc = wc(st.session_state.get("variation_text_2", ""))
        if v2_wc:
            delta = v2_wc - orig_wc_for_delta if orig_wc_for_delta else 0
            delta_str = f" ({'+' if delta >= 0 else ''}{delta} vs original)" if orig_wc_for_delta else ""
            st.caption(f"{v2_wc} words{delta_str}")
        copy_button(st.session_state.get("variation_text_2", ""), key="v2")

    st.markdown("---")

    # ── Target keyword ─────────────────────────────────────────────────────────
    st.text_input(
        "Target SEO keyword (optional — suggests which entity to optimize for)",
        key="target_keyword", placeholder="e.g. luxury spa Georgia",
    )
    if st.session_state.get("keyword_suggestion"):
        st.info(
            f"💡 Suggested entity for **\"{st.session_state['target_keyword']}\"**: "
            f"**{st.session_state['keyword_suggestion']}**"
        )

    # ── Controls row ───────────────────────────────────────────────────────────
    ctrl_left, ctrl_right = st.columns([5, 1])
    with ctrl_left:
        st.radio("Generate with Claude for:", ["Variation 1", "Variation 2"],
                 horizontal=True, key="assign_to")
    with ctrl_right:
        st.markdown('<div style="padding-top:28px;">', unsafe_allow_html=True)
        st.button("Clear all", on_click=clear_all)
        st.markdown('</div>', unsafe_allow_html=True)

    col_gen, col_analyze, col_suggest = st.columns([2, 2, 2])

    with col_gen:
        st.button(
            "Generate with Claude",
            on_click=on_generate_with_claude,
            disabled=not bool(ANTHROPIC_API_KEY),
        )

    if st.session_state.get("claude_error"):
        st.error(st.session_state["claude_error"])
    if not ANTHROPIC_API_KEY:
        st.warning("ANTHROPIC_API_KEY is not set. Add it to .env and restart.")

    # ── Analyze ────────────────────────────────────────────────────────────────
    original_text = st.session_state.get("original_text", "")
    variation_text_1 = st.session_state.get("variation_text_1", "")
    variation_text_2 = st.session_state.get("variation_text_2", "")

    with col_analyze:
        analyze_clicked = st.button("Analyze")

    with col_suggest:
        suggest_clicked = st.button(
            "Suggest entity",
            disabled=not bool(ANTHROPIC_API_KEY) or not st.session_state.get("target_keyword", "").strip(),
            help="Requires a target keyword and at least one Analyze run.",
        )

    if suggest_clicked:
        df = st.session_state.get("display_df")
        kw = st.session_state.get("target_keyword", "").strip()
        if df is None:
            st.warning("Run Analyze first.")
        elif not kw:
            st.warning("Enter a target keyword first.")
        else:
            with st.spinner("Asking Claude…"):
                try:
                    suggestion = suggest_entity_for_keyword(df["Entity"].tolist(), kw)
                    st.session_state["keyword_suggestion"] = suggestion
                    st.rerun()
                except Exception as e:
                    st.error(f"Suggestion failed: {e}")

    if analyze_clicked:
        v1_source_label = st.session_state.get("variation_source_1", "Pasted text")
        v2_source_label = st.session_state.get("variation_source_2", "Pasted text")

        if not any([original_text.strip(), variation_text_1.strip(), variation_text_2.strip()]):
            st.error("Add some text to analyze first — paste text in a variation box or click 'Analyze this element' to set an original.")
        else:
            with st.spinner("Scoring with Google NLP…"):
                all_entities = {}
                if original_text.strip():
                    all_entities["Original"] = analyze_text_salience(original_text)
                if variation_text_1.strip():
                    all_entities["Variation 1"] = analyze_text_salience(variation_text_1)
                    st.session_state["analyze_v1_label"] = v1_source_label
                else:
                    st.session_state["analyze_v1_label"] = ""
                if variation_text_2.strip():
                    all_entities["Variation 2"] = analyze_text_salience(variation_text_2)
                    st.session_state["analyze_v2_label"] = v2_source_label
                else:
                    st.session_state["analyze_v2_label"] = ""

            rows_list = []
            unique_entities = set(e for ents in all_entities.values() for e in ents)
            for entity in unique_entities:
                row = {"Entity": entity, "Type": None,
                       "Original": None, "Variation 1": None, "Variation 2": None}
                salience_scores = []
                for version, ents in all_entities.items():
                    if entity in ents:
                        s = float(ents[entity]["Salience"])
                        row["Type"] = ents[entity]["Type"]
                        row[version] = s
                        salience_scores.append(s)
                row["Average Salience"] = float(np.mean(salience_scores)) if salience_scores else np.nan
                rows_list.append(row)

            comparison_df = pd.DataFrame(rows_list)

            if comparison_df.empty:
                st.session_state["display_df"] = None
                st.warning(
                    "No entities were found. This usually means the text is too short or too generic. "
                    "Try pasting a longer passage with named people, places, or organizations."
                )
            else:
                comparison_df = comparison_df.sort_values(by="Average Salience", ascending=False)
                new_df = comparison_df.drop(columns=["Average Salience"])
                st.session_state["display_df"] = new_df

                # Save to session history (newest first, max 10)
                st.session_state["session_history"] = ([{
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "url": st.session_state.get("page_url", "") or "pasted text",
                    "entity_1": st.session_state.get("target_entity_1", ""),
                    "entity_2": st.session_state.get("target_entity_2", ""),
                    "df": new_df.copy(),
                }] + st.session_state.get("session_history", []))[:10]

    # ── Results ────────────────────────────────────────────────────────────────
    if st.session_state.get("display_df") is not None:
        display_df = st.session_state["display_df"]

        # Winner highlight
        winner_lines = []
        for col_name, entity_key in [("Variation 1", "target_entity_1"), ("Variation 2", "target_entity_2")]:
            entity = st.session_state.get(entity_key, "").strip()
            if not entity:
                continue
            match = display_df[display_df["Entity"].str.lower() == entity.lower()]
            if match.empty:
                continue
            row = match.iloc[0]
            score_cols_present = [c for c in ["Original", "Variation 1", "Variation 2"]
                                   if c in row.index and pd.notna(row[c])]
            if not score_cols_present:
                continue
            best_col = max(score_cols_present, key=lambda c: row[c])
            best_score = row[best_col]
            orig_score = row.get("Original") if "Original" in row.index and pd.notna(row.get("Original")) else None
            improvement = (
                f" — up from <strong>{orig_score:.2f}</strong> original"
                if orig_score and best_col != "Original" else ""
            )
            color = score_color(best_score)
            winner_lines.append(
                f'<span style="color:{color};font-weight:600">▲ {entity}</span>: '
                f'best in <strong>{best_col}</strong> at '
                f'<span style="color:{color};font-weight:600">{best_score:.2f}</span>{improvement}'
            )

        if winner_lines:
            st.markdown(
                '<div style="background:rgba(255,255,255,0.04);border-radius:6px;'
                'padding:10px 14px;font-size:13px;margin-bottom:4px;">'
                + " &nbsp;|&nbsp; ".join(winner_lines) + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(SCORE_GUIDE, unsafe_allow_html=True)

        # Build display table with element labels in column headers
        v1_lbl = st.session_state.get("analyze_v1_label", "")
        v2_lbl = st.session_state.get("analyze_v2_label", "")
        table_df = display_df.copy()
        col_rename = {}
        if v1_lbl and "Variation 1" in table_df.columns:
            col_rename["Variation 1"] = f"Variation 1 — {v1_lbl}"
        if v2_lbl and "Variation 2" in table_df.columns:
            col_rename["Variation 2"] = f"Variation 2 — {v2_lbl}"
        if col_rename:
            table_df = table_df.rename(columns=col_rename)

        score_cols = [c for c in table_df.columns if c in
                      ["Original"] + list(col_rename.values()) +
                      (["Variation 1"] if "Variation 1" in table_df.columns else []) +
                      (["Variation 2"] if "Variation 2" in table_df.columns else [])]
        score_cols = [c for c in table_df.columns if c not in ("Entity", "Type")]
        styled = table_df.style.map(style_score, subset=score_cols)

        st.dataframe(
            styled,
            width="stretch",
            hide_index=True,
            on_select=assign_selected_entity,
            selection_mode="single-row",
            key="entity_table",
            column_config={c: st.column_config.NumberColumn(format="%.2f") for c in score_cols},
        )

        export_df = display_df.copy()
        page_url = st.session_state.get("page_url", "")
        export_df.insert(0, "URL", page_url if page_url.strip() else "—")
        st.download_button(
            "Export to CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="salience_analysis.csv",
            mime="text/csv",
        )

    # ── Session history ────────────────────────────────────────────────────────
    history = st.session_state.get("session_history", [])
    if len(history) > 1:
        with st.expander(f"Session history ({len(history)} analyses)"):
            for i, entry in enumerate(history):
                url_label = entry["url"] if entry["url"] != "pasted text" else "Pasted text"
                entities_label = " / ".join(
                    filter(None, [entry.get("entity_1"), entry.get("entity_2")])
                ) or "—"
                st.markdown(f"**{entry['time']}** — {url_label} — _{entities_label}_")
                h_df = entry["df"]
                h_score_cols = [c for c in ["Original", "Variation 1", "Variation 2"] if c in h_df.columns]
                h_styled = h_df.style.map(style_score, subset=h_score_cols)
                st.dataframe(
                    h_styled,
                    width="stretch",
                    hide_index=True,
                    key=f"hist_{i}",
                    column_config={
                        "Original": st.column_config.NumberColumn(format="%.2f"),
                        "Variation 1": st.column_config.NumberColumn(format="%.2f"),
                        "Variation 2": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
                if i < len(history) - 1:
                    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# BULK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab_bulk:
    st.markdown("### Bulk URL Analysis")
    st.markdown(
        "Paste a list of URLs (one per line). The tool fetches each page, scores the H1 and first "
        "sentence, and returns the top entity and salience score for each. Optionally enter a target "
        "entity to see its specific score across all pages."
    )

    st.text_area("URLs (one per line):", height=180, key="bulk_urls_input",
                 placeholder="https://example.com/page-1\nhttps://example.com/page-2")
    st.text_input("Target entity (optional — scored across all URLs):",
                  key="bulk_target_entity", placeholder="e.g. Napa Valley")

    if st.button("Run Bulk Analysis", key="run_bulk"):
        urls = [u.strip() for u in st.session_state["bulk_urls_input"].splitlines() if u.strip()]
        if not urls:
            st.error("Paste at least one URL.")
        else:
            results = []
            progress_bar = st.progress(0, text="Starting…")

            for i, url in enumerate(urls):
                progress_bar.progress(i / len(urls), text=f"Processing {i + 1}/{len(urls)}: {url[:70]}…")
                try:
                    data = fetch_page_elements(url)
                    h1 = data.get("h1", "")
                    fs = data.get("first_sentence_after_h1", "")
                    target = st.session_state.get("bulk_target_entity", "").strip()

                    h1_entity, h1_score = (None, None)
                    if h1.strip():
                        h1_ents = analyze_text_salience(h1)
                        h1_entity, h1_score = top_entity(h1_ents)
                        target_h1_score = h1_ents.get(target, {}).get("Salience") if target else None
                    else:
                        h1_ents = {}
                        target_h1_score = None

                    fs_entity, fs_score = (None, None)
                    if fs.strip():
                        fs_ents = analyze_text_salience(fs)
                        fs_entity, fs_score = top_entity(fs_ents)
                        target_fs_score = fs_ents.get(target, {}).get("Salience") if target else None
                    else:
                        target_fs_score = None

                    row = {
                        "URL": url,
                        "H1": h1,
                        "H1 Top Entity": h1_entity or "—",
                        "H1 Top Score": round(h1_score, 2) if h1_score is not None else None,
                        "First Sentence Top Entity": fs_entity or "—",
                        "First Sentence Top Score": round(fs_score, 2) if fs_score is not None else None,
                        "Error": "",
                    }
                    if target:
                        row["H1 Target Score"] = round(target_h1_score, 2) if target_h1_score is not None else None
                        row["FS Target Score"] = round(target_fs_score, 2) if target_fs_score is not None else None

                    results.append(row)

                except Exception as e:
                    results.append({
                        "URL": url, "H1": "", "H1 Top Entity": "—", "H1 Top Score": None,
                        "First Sentence Top Entity": "—", "First Sentence Top Score": None,
                        "Error": str(e),
                    })

            progress_bar.progress(1.0, text=f"Done! Processed {len(urls)} URLs.")
            st.session_state["bulk_results_df"] = pd.DataFrame(results)

    if st.session_state.get("bulk_results_df") is not None:
        bulk_df = st.session_state["bulk_results_df"]
        # Hide Error column when there are no errors
        bulk_display = bulk_df.copy()
        if "Error" in bulk_display.columns and bulk_display["Error"].fillna("").eq("").all():
            bulk_display = bulk_display.drop(columns=["Error"])
        bulk_score_cols = [c for c in bulk_display.columns if "Score" in c]
        bulk_styled = bulk_display.style.map(style_score, subset=bulk_score_cols)

        st.dataframe(
            bulk_styled,
            width="stretch",
            hide_index=True,
            column_config={c: st.column_config.NumberColumn(format="%.2f") for c in bulk_score_cols},
        )

        st.download_button(
            "Export Bulk Results to CSV",
            data=bulk_display.to_csv(index=False).encode("utf-8"),
            file_name="bulk_salience.csv",
            mime="text/csv",
        )
