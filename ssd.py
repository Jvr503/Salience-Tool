# Script created by Javier Hernandez to show the salience score of entities extracted from a text, using Google NLP API.

import os
import streamlit as st
from google.cloud import language_v1
import pandas as pd
import numpy as np

# Set the environment variable to the path of your Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/javier/Salience/gen-lang-client-0125666263-e9763df6c096.json"

# --- Inject Propellic Branding Styles ---
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }
    .stButton>button {
        background-color: #E21A6B;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.4em;
    }
    .stButton>button:hover {
        background-color: #c0175d;
    }
    </style>
""", unsafe_allow_html=True)

# --- Propellic Logo ---
st.image("propellic-logo-png.png", width=180)

# --- Header with eyebrow and title ---
st.markdown("""
    <div style="text-transform:uppercase; color:#E21A6B; font-weight:bold; font-size:14px; margin-bottom:0.5rem;">Salience Analyzer</div>
    <h1 style='color:white; margin-top:0;'>Text Analysis with Google NLP</h1>
""", unsafe_allow_html=True)

# --- Text Inputs ---
original_text = st.text_area("Paste the original content you want to analyze:", height=100)
variation_text_1 = st.text_area("Paste the content for Variation 1 (optional):", height=100)
variation_text_2 = st.text_area("Paste the content for Variation 2 (optional):", height=100)

def analyze_text_salience(text):
    """Analyzes the text and returns entities with their salience scores."""
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_entities(document=document)
    entity_dict = {}
    for entity in response.entities:
        entity_dict[entity.name] = {
            "Type": language_v1.Entity.Type(entity.type_).name,
            "Salience": entity.salience
        }
    return entity_dict

# --- Analyze Button ---
if st.button('Analyze'):
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
                salience_score = round(entities[entity]["Salience"], 2)
                row["Type"] = entities[entity]["Type"]
                row[text_version] = salience_score
                salience_scores.append(salience_score)
        row["Average Salience"] = round(np.mean(salience_scores), 2) if salience_scores else None
        rows_list.append(row)

    comparison_df = pd.DataFrame(rows_list)
    comparison_df = comparison_df.sort_values(by="Average Salience", ascending=False)
    comparison_df = comparison_df.drop(columns=["Average Salience"])
    
    for col in ["Original", "Variation 1", "Variation 2"]:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    
    for col in ["Original", "Variation 1", "Variation 2", "Average Salience"]:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)

    comparison_df = comparison_df.fillna("")

    if not comparison_df.empty:
        st.table(comparison_df)
    else:
        st.write("No entities found or no text provided.")
