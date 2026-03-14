# Script created by Javier Hernandez to show the salience score of entities extracted from a text, using Google NLP API.

import io
import os
import re
import requests
import pandas as pd
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.cloud import language_v1
import anthropic

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")

# -----------------------------
# Helper: first sentence
# -----------------------------
def first_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return parts[0].strip()

# -----------------------------
# Helper: fetch URL elements
# -----------------------------
def fetch_page_elements(url: str) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
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

# -----------------------------
# Credentials checks
# -----------------------------
cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if not cred:
    st.error("GOOGLE_APPLICATION_CREDENTIALS is not set. Add it to ~/.zprofile or .env, then restart the app.")
    st.stop()

cred_path = os.path.expanduser(cred)
if not os.path.exists(cred_path):
    st.error(f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {cred}")
    st.stop()

# -----------------------------
# Branding / UI
# -----------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    .stButton>button {
        background-color: #E21A6B;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.4em;
    }
    .stButton>button:hover { background-color: #c0175d; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.image("propellic-logo-png.png", width=180)

st.markdown(
    """
    <div style="text-transform:uppercase; color:#E21A6B; font-weight:bold; font-size:14px; margin-bottom:0.5rem;">
      Salience Analyzer
    </div>
    <h1 style='color:white; margin-top:0;'>Text Analysis with Google NLP</h1>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Claude helpers
# -----------------------------
@st.cache_resource
def get_claude_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)

# Short elements (H1, meta title) — stay as headings, no expansion
_SHORT_ELEMENTS = {"H1", "Meta title"}

def generate_optimized_text(original: str, target_entity: str, source_label: str = "Pasted text") -> str:
    client = get_claude_client(ANTHROPIC_API_KEY)

    if source_label in _SHORT_ELEMENTS:
        word_count = len(original.split())
        prompt = f"""You are optimizing a {source_label} for Google Cloud Natural Language API entity salience.

Goal: Rewrite this {source_label} so that "{target_entity}" is the #1 entity by salience.

Rules:
1) This is a {source_label} — keep it as a short heading. Do NOT write a sentence or paragraph.
2) Keep the output at roughly {word_count} words (±2 words maximum). Never expand it.
3) Place "{target_entity}" as the first or second word of the heading.
4) Remove or demote any competing named entities — replace them with generic descriptors if needed (e.g. "the spa", "the resort", "the winery").
5) Return ONLY the rewritten {source_label}. No punctuation at the end unless the original had it. No explanation.

Original {source_label}:
{original}""".strip()
    else:
        prompt = f"""You are optimizing text for Google Cloud Natural Language API entity salience.

Goal: When analyzed with Google NLP analyze_entities, the entity named exactly "{target_entity}" must be the #1 entity by salience, and it should clearly dominate any other entity.

Rewrite rules:
1) Put "{target_entity}" in the first sentence, within the first 8–10 words.
2) Make "{target_entity}" the grammatical subject of most sentences (active voice). Avoid making any other named entity the subject.
3) Use the exact string "{target_entity}" multiple times naturally — do not replace all mentions with pronouns or synonyms.
4) Demote competing entities: keep other named entities to a minimum. If you must keep one from the original, rewrite it as a generic reference (e.g. "the resort", "the region", "the tasting room") after the first mention.
5) Do not introduce any new named entities that were not already in the original text.
6) Keep the rewrite the same length or shorter — do NOT expand the original text. Tighter, more focused writing scores better.

Keep meaning and tone. Keep it natural and readable.

Output: Return ONLY the rewritten text. No headings. No explanation. No extra formatting. No **.

Original text:
{original}""".strip()

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()

# -----------------------------
# Session state defaults
# -----------------------------
defaults = {
    "url_meta_title": "",
    "url_meta_description": "",
    "url_h1": "",
    "url_first_sentence": "",
    "claude_error": "",
    "page_url": "",
    "original_text": "",
    "variation_text_1": "",
    "variation_text_2": "",
    "target_entity_1": "",
    "target_entity_2": "",
    "variation_source_1": "Pasted text",
    "variation_source_2": "Pasted text",
    "assign_to": "Variation 1",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.get("assign_to") not in ["Variation 1", "Variation 2"]:
    st.session_state["assign_to"] = "Variation 1"

# -----------------------------
# Inputs (tabs) — original text only
# -----------------------------
tab_paste, tab_url = st.tabs(["Paste text", "Load from URL"])

with tab_url:
    st.text_input("Page URL", key="page_url")

    if st.button("Load page elements", key="load_page_elements"):
        if not st.session_state["page_url"].strip():
            st.error("Paste a URL first.")
        else:
            try:
                data = fetch_page_elements(st.session_state["page_url"].strip())

                st.session_state["url_meta_title"] = data.get("meta_title", "")
                st.session_state["url_meta_description"] = data.get("meta_description", "")
                st.session_state["url_h1"] = data.get("h1", "")
                st.session_state["url_first_sentence"] = data.get("first_sentence_after_h1", "")
                st.session_state["original_text"] = data.get("prefill_original", "")

                st.rerun()
            except Exception as e:
                st.error(f"Failed to fetch URL: {e}")

    if "url_meta_title_display" not in st.session_state:
        st.session_state["url_meta_title_display"] = ""
    if "url_meta_description_display" not in st.session_state:
        st.session_state["url_meta_description_display"] = ""
    if "url_h1_display" not in st.session_state:
        st.session_state["url_h1_display"] = ""
    if "url_first_sentence_display" not in st.session_state:
        st.session_state["url_first_sentence_display"] = ""

    st.session_state["url_meta_title_display"] = st.session_state["url_meta_title"]
    st.session_state["url_meta_description_display"] = st.session_state["url_meta_description"]
    st.session_state["url_h1_display"] = st.session_state["url_h1"]
    st.session_state["url_first_sentence_display"] = st.session_state["url_first_sentence"]

    st.text_input("Meta title", key="url_meta_title_display", disabled=True)
    st.text_area("Meta description", key="url_meta_description_display", height=80, disabled=True)
    st.text_input("H1", key="url_h1_display", disabled=True)
    st.text_area("First sentence after H1", key="url_first_sentence_display", height=80, disabled=True)

with tab_paste:
    st.text_area(
        "Paste the original content you want to analyze:",
        height=100,
        key="original_text",
    )

# Source options for the element selector
_SOURCE_OPTIONS = ["Pasted text", "H1", "First sentence after H1", "Meta title", "Meta description"]
_SOURCE_KEY_MAP = {
    "Pasted text": "original_text",
    "H1": "url_h1",
    "First sentence after H1": "url_first_sentence",
    "Meta title": "url_meta_title",
    "Meta description": "url_meta_description",
}

# -----------------------------
# Variation 1 — clearly grouped
# -----------------------------
st.markdown("---")
with st.container(border=True):
    st.markdown("#### Variation 1")
    st.text_input(
        "Target entity for Variation 1",
        key="target_entity_1",
        placeholder="e.g. Napa Valley",
    )
    st.selectbox(
        "Element to optimize",
        options=_SOURCE_OPTIONS,
        key="variation_source_1",
        help="Choose which page element Claude should rewrite. URL elements are available after loading a URL above.",
    )
    st.text_area(
        "Variation 1 text (paste manually or click Generate below):",
        height=100,
        key="variation_text_1",
    )

# -----------------------------
# Variation 2 — clearly grouped
# -----------------------------
with st.container(border=True):
    st.markdown("#### Variation 2")
    st.text_input(
        "Target entity for Variation 2",
        key="target_entity_2",
        placeholder="e.g. Sonoma County",
    )
    st.selectbox(
        "Element to optimize",
        options=_SOURCE_OPTIONS,
        key="variation_source_2",
        help="Choose which page element Claude should rewrite. URL elements are available after loading a URL above.",
    )
    st.text_area(
        "Variation 2 text (paste manually or click Generate below):",
        height=100,
        key="variation_text_2",
    )

st.markdown("---")

# -----------------------------
# Generate + Analyze controls
# -----------------------------
st.radio(
    "Generate with Claude for:",
    ["Variation 1", "Variation 2"],
    horizontal=True,
    key="assign_to",
)

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
        st.session_state["claude_error"] = "Pick or type a target entity first."
        return

    st.session_state["claude_error"] = ""

    try:
        new_text = generate_optimized_text(original, target, source_label)
    except Exception as e:
        st.session_state["claude_error"] = f"Claude request failed: {e}"
        return

    if st.session_state["assign_to"] == "Variation 1":
        st.session_state["variation_text_1"] = new_text
    else:
        st.session_state["variation_text_2"] = new_text

col_gen, col_analyze = st.columns([1, 1])

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

# Keep these variables available for the rest of the script
original_text = st.session_state.get("original_text", "")
variation_text_1 = st.session_state.get("variation_text_1", "")
variation_text_2 = st.session_state.get("variation_text_2", "")

# -----------------------------
# Google NLP
# -----------------------------
def analyze_text_salience(text: str) -> dict:
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_entities(document=document)

    entity_dict = {}
    for entity in response.entities:
        entity_dict[entity.name] = {
            "Type": language_v1.Entity.Type(entity.type_).name,
            "Salience": float(entity.salience),
        }
    return entity_dict

# -----------------------------
# Analyze + table
# -----------------------------
if "display_df" not in st.session_state:
    st.session_state["display_df"] = None

def _score_color(val: float) -> str:
    if val >= 0.60: return "#16a34a"
    if val >= 0.40: return "#ca8a04"
    if val >= 0.20: return "#ea580c"
    return "#dc2626"

def _score_cell(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '<td class="sc-num">—</td>'
    color = _score_color(val)
    return f'<td class="sc-num"><span style="color:{color}">●</span> {val:.2f}</td>'

def build_score_table(df: pd.DataFrame) -> str:
    score_cols = {"Original", "Variation 1", "Variation 2"}
    headers = "".join(
        f'<th class="sc-h sc-num">{c}</th>' if c in score_cols else f'<th class="sc-h">{c}</th>'
        for c in df.columns
    )
    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        bg = "rgba(255,255,255,0.03)" if i % 2 == 0 else "transparent"
        cells = ""
        for col in df.columns:
            if col in score_cols:
                cells += _score_cell(row[col])
            else:
                val = row[col] if row[col] is not None else "—"
                cells += f'<td class="sc-td">{val}</td>'
        rows_html += f'<tr style="background:{bg}">{cells}</tr>'
    return f"""
<style>
  .sc-table {{width:100%;border-collapse:collapse;font-size:13px;font-family:'Montserrat',sans-serif;}}
  .sc-h {{text-align:left;padding:8px 12px;border-bottom:1px solid rgba(255,255,255,0.12);
          color:rgba(255,255,255,0.55);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;}}
  .sc-h.sc-num {{text-align:right;}}
  .sc-td {{padding:7px 12px;border-bottom:1px solid rgba(255,255,255,0.05);color:rgba(255,255,255,0.9);}}
  .sc-num {{text-align:right;padding:7px 12px;border-bottom:1px solid rgba(255,255,255,0.05);
            color:rgba(255,255,255,0.9);font-variant-numeric:tabular-nums;}}
</style>
<table class="sc-table"><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table>
"""

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

with col_analyze:
    analyze_clicked = st.button("Analyze")

if analyze_clicked:
    all_entities = {}

    if original_text:
        all_entities["Original"] = analyze_text_salience(original_text)
    if variation_text_1:
        all_entities["Variation 1"] = analyze_text_salience(variation_text_1)
    if variation_text_2:
        all_entities["Variation 2"] = analyze_text_salience(variation_text_2)

    rows_list = []
    unique_entities = set(entity for text in all_entities.values() for entity in text)

    for entity in unique_entities:
        row = {"Entity": entity, "Type": None, "Original": None, "Variation 1": None, "Variation 2": None}
        salience_scores = []

        for text_version, entities in all_entities.items():
            if entity in entities:
                salience_score = float(entities[entity]["Salience"])
                row["Type"] = entities[entity]["Type"]
                row[text_version] = salience_score
                salience_scores.append(salience_score)

        row["Average Salience"] = float(np.mean(salience_scores)) if salience_scores else np.nan
        rows_list.append(row)

    comparison_df = pd.DataFrame(rows_list)

    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(by="Average Salience", ascending=False)
        st.session_state["display_df"] = comparison_df.drop(columns=["Average Salience"])
    else:
        st.session_state["display_df"] = None
        st.write("No entities found or no text provided.")

# -----------------------------
# Score guide + results table + export
# -----------------------------
if st.session_state.get("display_df") is not None:
    st.markdown(
        """
        <div style="background:rgba(226,26,107,0.08); border-left:4px solid #E21A6B;
                    padding:10px 14px; border-radius:4px; font-size:13px; margin:12px 0;">
          <strong>Salience Score Guide</strong> &nbsp;—&nbsp;
          <span style="color:#16a34a; font-weight:600">0.60+</span>&nbsp;Excellent: entity clearly dominates &nbsp;
          <span style="color:#ca8a04; font-weight:600">0.40–0.59</span>&nbsp;Good: strong signal &nbsp;
          <span style="color:#ea580c; font-weight:600">0.20–0.39</span>&nbsp;Moderate: competing entities dilute focus &nbsp;
          <span style="color:#dc2626; font-weight:600">&lt; 0.20</span>&nbsp;Weak: entity is not the clear subject
        </div>
        """,
        unsafe_allow_html=True,
    )

    display_df = st.session_state["display_df"]

    def style_score(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        if val >= 0.60: return "color: #16a34a; font-weight: 600"
        if val >= 0.40: return "color: #ca8a04; font-weight: 600"
        if val >= 0.20: return "color: #ea580c; font-weight: 600"
        return "color: #dc2626; font-weight: 600"

    score_cols = [c for c in ["Original", "Variation 1", "Variation 2"] if c in display_df.columns]
    styled = display_df.style.map(style_score, subset=score_cols)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select=assign_selected_entity,
        selection_mode="single-row",
        key="entity_table",
        column_config={
            "Original": st.column_config.NumberColumn(format="%.2f"),
            "Variation 1": st.column_config.NumberColumn(format="%.2f"),
            "Variation 2": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    export_df = display_df.copy()
    page_url = st.session_state.get("page_url", "")
    export_df.insert(0, "URL", page_url if page_url.strip() else "—")

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export to CSV",
        data=csv_bytes,
        file_name="salience_analysis.csv",
        mime="text/csv",
    )
