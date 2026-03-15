# Salience Tool

A Propellic internal tool for analyzing and optimizing entity salience scores using the Google Cloud Natural Language API. Paste or load page content, compare up to three versions of a text, generate AI-optimized rewrites with Claude, and export results to CSV.

---

## What It Does

Google's Natural Language API assigns each named entity in a piece of text a **salience score** (0–1) that measures how central that entity is as the subject of the text. A score of 0.60+ means the entity clearly dominates; below 0.20 means it barely registers.

This tool lets you:
- Score the original text from any URL or pasted copy
- Generate Claude-optimized rewrites that push a target entity to #1 salience
- Compare Original, Variation 1, and Variation 2 side by side in a sortable table
- Export results to CSV for reporting

**Salience Score Guide**

| Score | Meaning |
|-------|---------|
| 0.60+ | Excellent — entity clearly dominates |
| 0.40–0.59 | Good — strong signal, room to improve |
| 0.20–0.39 | Moderate — competing entities dilute focus |
| < 0.20 | Weak — entity is not the clear subject |

---

## Requirements

- Python 3.10+
- A Google Cloud project with the **Natural Language API** enabled
- A Google Cloud service account key JSON file
- An **Anthropic API key** (for Claude-powered rewrites)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

### 1. Google Cloud credentials

Place your service account key JSON file in the project folder (it is git-ignored). The file is named `gen-lang-client-*.json`.

### 2. Environment variables

Create a `.env` file in the project root (also git-ignored):

```
ANTHROPIC_API_KEY=sk-ant-...
```

The `GOOGLE_APPLICATION_CREDENTIALS` path is set automatically by `run-salience.sh`. If running manually, export it yourself:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gen-lang-client-*.json"
```

---

## How to Run

### Recommended — use the launch script:

```bash
./run-salience.sh
```

This activates the virtual environment, sets credentials, and starts Streamlit in one step.

### Manual:

```bash
source venv314/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gen-lang-client-*.json"
streamlit run ssd.py
```

The app opens at `http://localhost:8501` (or the next available port).

---

## How to Use

### Step 1 — Load content
- **Paste text tab**: Paste the original copy you want to analyze directly.
- **Load from URL tab**: Enter a page URL and click "Load page elements". The tool extracts the H1 and first sentence after it and pre-fills the original text box. It also shows the meta title and meta description for reference.

### Step 2 — Set up variations
Two bordered sections appear below the tabs — one for each variation:

- **Target entity**: Type the entity you want to make dominant (e.g. "Napa Valley").
- **Element to optimize**: Choose which page element Claude should rewrite — H1, First sentence after H1, Meta title, Meta description, or Pasted text. URL elements are only available after loading a URL.
- **Variation text**: Paste your own rewrite manually, or leave it empty and use Claude to generate one.

### Step 3 — Generate with Claude (optional)
1. Select which variation to generate using the **"Generate with Claude for"** radio buttons
2. Click **"Generate with Claude"**

Claude rewrites the selected element to make the target entity the dominant subject. The behavior adapts based on element type:
- **H1 / Meta title**: Stays as a short heading — same word count, entity placed first, no expansion.
- **First sentence / Meta description / Pasted text**: Full paragraph rewrite — entity as grammatical subject, competing entities demoted, same length or shorter.

### Step 4 — Analyze
Click **"Analyze"** to call the Google NLP API and score all provided texts. A table shows every entity with its salience score across Original, Variation 1, and Variation 2. Scores are color-coded:

| Color | Range | Meaning |
|-------|-------|---------|
| Green | 0.60+ | Excellent — entity clearly dominates |
| Yellow | 0.40–0.59 | Good — strong signal |
| Orange | 0.20–0.39 | Moderate — competing entities dilute focus |
| Red | < 0.20 | Weak — entity is not the clear subject |

Click any row in the table to assign that entity as the target for the currently selected variation.

### Step 5 — Export
Click **"Export to CSV"** to download the full results table (raw numeric scores) with the source URL prepended.

---

## Project Structure

```
├── ssd.py                          # Main Streamlit application
├── run-salience.sh                 # Launch script (activates venv + sets credentials)
├── requirements.txt                # Python dependencies
├── propellic-logo-png.png          # Branding asset
├── ssd_WORKING_BACKUP.py           # Pre-refactor backup
├── .env                            # API keys (git-ignored)
├── gen-lang-client-*.json          # Google Cloud key (git-ignored)
├── .gitignore
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit>=1.35.0` | UI framework |
| `google-cloud-language` | Entity salience scoring via Google NLP API |
| `anthropic` | Claude API for AI-powered rewrites |
| `beautifulsoup4` | HTML parsing for URL loading |
| `requests` | HTTP fetching |
| `pandas` / `numpy` | Data handling and table display |
| `python-dotenv` | `.env` file loading |

---

## Author

Created by Javier Hernandez — Propellic
