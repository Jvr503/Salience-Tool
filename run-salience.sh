#!/bin/zsh

cd ~/Salience-Tool
source venv314/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS="/Users/javierhernandez/Salience-Tool/gen-lang-client-0125666263-2268bc579fa0.json"
streamlit run ssd.py
