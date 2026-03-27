#!/bin/zsh

cd ~/Salience-Tool
source venv314/bin/activate

set -a
source .env
set +a

streamlit run ssd.py
