# Script created by Javier Hernandez to show the salience score of entities extracted from a text, using Google NLP API.

import os
import streamlit as st
from google.cloud import language_v1
import pandas as pd
import numpy as np

# Set the environment variable to the path of your Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/javier/Salience/gen-lang-client-0125666263-e9763df6c096.json"

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

# Streamlit web interface
st.title('Text Analysis with Google NLP')
original_text = st.text_area("Paste the original content you want to analyze:", height=100)
variation_text_1 = st.text_area("Paste the content for Variation 1 (optional):", height=100)
variation_text_2 = st.text_area("Paste the content for Variation 2 (optional):", height=100)

if st.button('Analyze'):
    all_entities = {}
    # Analyze each text and collect entities
    if original_text:
        all_entities["Original"] = analyze_text_salience(original_text)
    if variation_text_1:
        all_entities["Variation 1"] = analyze_text_salience(variation_text_1)
    if variation_text_2:
        all_entities["Variation 2"] = analyze_text_salience(variation_text_2)
    
    rows_list = []  # List to hold all rows
    unique_entities = set(entity for text in all_entities.values() for entity in text)
    
    for entity in unique_entities:
        row = {"Entity": entity, "Type": None, "Original": None, "Variation 1": None, "Variation 2": None}
        salience_scores = []  # Collect salience scores for calculating the average
        for text_version, entities in all_entities.items():
            if entity in entities:
                # Round the salience score to two decimal places before adding it to the row
                salience_score = round(entities[entity]["Salience"], 2)
                row["Type"] = entities[entity]["Type"]
                row[text_version] = salience_score  # Store the rounded score
                salience_scores.append(salience_score)
        # Calculate the average and round it to two decimal places
        row["Average Salience"] = round(np.mean(salience_scores), 2) if salience_scores else None
        rows_list.append(row)
    
    # Create DataFrame from the list of rows
    comparison_df = pd.DataFrame(rows_list)
    
    # Sort the DataFrame based on the 'Average Salience' column in descending order
    comparison_df = comparison_df.sort_values(by="Average Salience", ascending=False)
    
    # Optionally, you can drop the 'Average Salience' column if you don't want to display it
    comparison_df = comparison_df.drop(columns=["Average Salience"])
    
    # Convert all numerical values to strings to ensure consistent data types across the DataFrame
    for col in ["Original", "Variation 1", "Variation 2"]:
        if col in comparison_df.columns:  # Check if the column exists to avoid KeyErrors
            # Convert to string with two decimal places, but leave NaN/None as an empty string
            comparison_df[col] = comparison_df[col].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    # Convert all numerical values to strings to ensure consistent data types across the DataFrame
    for col in ["Original", "Variation 1", "Variation 2", "Average Salience"]:
        if col in comparison_df.columns:  # Check if the column exists to avoid KeyErrors
            comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
    
    # Fill NA/None values with an empty string
    comparison_df = comparison_df.fillna("")
    
    # Display the comparison table
    if not comparison_df.empty:
        st.table(comparison_df)
    else:
        st.write("No entities found or no text provided.")
