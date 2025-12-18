# Script created by Javier Hernandez to show the salience score of entities extracted from a text, using Google NLP API.

import os
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google.cloud import language_v1
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
if not GEMINI_CLIENT:
    st.warning("GEMINI_API_KEY is not set. Gemini generation will be disabled.")

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
# Gemini helpers
# -----------------------------
@st.cache_resource
def get_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def generate_optimized_text(original: str, target_entity: str) -> str:
    client = get_gemini_client(GEMINI_API_KEY)

    prompt = f"""

You are optimizing text for Google Cloud Natural Language API entity salience.

Goal: When analyzed with Google NLP analyze_entities, the entity named exactly "{target_entity}" must be the #1 entity by salience.

Rewrite rules:

Put "{target_entity}" in the first sentence, within the first 10 words.

Make "{target_entity}" the grammatical subject of most sentences (use active voice). Do not make other entities the subject.

Repeat the exact string "{target_entity}" at least 3 times, spread across the text. Do not replace all repeats with pronouns.

Minimize other named entities (people/places/brands/products/events). If you must keep them, demote them into prepositional phrases or subordinate clauses and refer to them generically (example: “the winery”, “the region”) instead of repeating the proper name.

Keep meaning and tone. Keep length roughly similar (within ±15%). Keep it natural and readable.

Output: Return ONLY the rewritten text. No headings. No explanation.

Original text:
{original}

"""

    r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)

    text = getattr(r, "text", None)
    return (text or "").strip()


# -----------------------------
# Inputs
# -----------------------------
original_text = st.text_area("Paste the original content you want to analyze:", height=100, key="original_text")
variation_text_1 = st.text_area("Paste the content for Variation 1 (optional):", height=100, key="variation_text_1")
variation_text_2 = st.text_area("Paste the content for Variation 2 (optional):", height=100, key="variation_text_2")

st.text_input("Target entity for Variation 1", key="target_entity_1")
st.text_input("Target entity for Variation 2", key="target_entity_2")

assign_to = st.radio(
    "When I click an entity in the table, assign it to:",
    ["Variation 1", "Variation 2"],
    horizontal=True,
    key="assign_to",
)

if "gemini_error" not in st.session_state:
    st.session_state["gemini_error"] = ""


def on_generate_with_gemini():
    # This runs before widgets render on rerun, so it's safe to set text_area keys here.
    if not GEMINI_API_KEY:
        st.session_state["gemini_error"] = "GEMINI_API_KEY is not set. Add it to .env and restart."
        return

    target = st.session_state["target_entity_1"] if st.session_state["assign_to"] == "Variation 1" else st.session_state["target_entity_2"]
    original = st.session_state.get("original_text", "")

    if not original.strip():
        st.session_state["gemini_error"] = "Paste original text first."
        return

    if not target.strip():
        st.session_state["gemini_error"] = "Pick or type a target entity first."
        return

    st.session_state["gemini_error"] = ""

    try:
        new_text = generate_optimized_text(original, target)
    except Exception as e:
        st.session_state["gemini_error"] = f"Gemini request failed: {e}"
        return

    if st.session_state["assign_to"] == "Variation 1":
        st.session_state["variation_text_1"] = new_text
    else:
        st.session_state["variation_text_2"] = new_text


st.button("Generate with Gemini", on_click=on_generate_with_gemini, disabled=not bool(GEMINI_API_KEY))

if st.session_state.get("gemini_error"):
    st.error(st.session_state["gemini_error"])

if not GEMINI_API_KEY:
    st.warning("GEMINI_API_KEY is not set. Gemini generation will be disabled.")


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


if st.button("Analyze"):
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


if st.session_state.get("display_df") is not None:
    display_df = st.session_state["display_df"]
    st.dataframe(
        display_df,
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
