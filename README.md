# Salience-Tool
The tool uses the Google Cloud Natural Language API to extract named entities from text and evaluate their salience score—which measures how important each entity is to the overall meaning of the text.

This app includes custom branding and UI styling for Propellic, along with a simple, intuitive interface built using Streamlit.

---

### Features

- Compare up to 3 versions of a text
- Extract and rank named entities by salience
- View salience scores in a sortable table
- Branded interface with Propellic logo and colors

---

### Requirements

Make sure you have Python 3.7+ installed, then install dependencies:

pip install -r requirements.txt


Also required:
- A Google Cloud project with the Natural Language API enabled
- A service account key JSON file downloaded from your project

---

### How to Run

1. Place your Google Cloud key file in the project folder (not tracked by Git)
2. Set the environment variable in your terminal:

export GOOGLE_APPLICATION_CREDENTIALS="gen-lang-client-xxxxxxxxxxxxx.json"

3. Start the app:

streamlit run ssd.py


The app will open in your browser at `http://localhost:8501`.

---

### Project Structure

.
├── ssd.py # Streamlit app
├── requirements.txt # Dependencies
├── propellic-logo-png.png # Custom branding
├── .gitignore # Ignores JSON keys and cache files
├── LICENSE
├── README.md
└── .streamlit/ # (Optional) Streamlit config folder


---

### Author

Created by Javier Hernandez  
Branded for Propellic

---

### License

This project is licensed under the MIT License.
