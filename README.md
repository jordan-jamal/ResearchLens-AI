# ResearchLens AI 🔬

**Faithful Research Paper Summarization with Evidence-Based Verification**

Powered by **Gemini 3.5 Flash-Lite** · Built with Streamlit

---

## Features

| Feature | Description |
|---|---|
| 📄 **PDF Extraction** | Extracts text from any research paper PDF |
| 🧠 **AI Summarization** | Structured 7-section summary (Problem → Conclusions) |
| ✅ **Faithfulness Check** | Verifies each summary claim against the original paper |
| 💬 **Ask the Paper** | Chat-style Q&A grounded in the paper's text |
| 📖 **Technical Glossary** | Extracts & defines key terms, acronyms, and concepts in plain English |
| 📚 **Citation Extractor** | Parses all references with authors, title, venue, year & DOI links |

---

## Local Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd ResearchPaperSummarizer

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key
echo GEMINI_API_KEY=your-key-here > .env

# 5. Run the app
streamlit run app.py
```

---

## Deploy to Streamlit Cloud

1. Push the project to a GitHub repository (the `venv/` folder is already git-ignored).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repo and set **Main file path** to `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-gemini-api-key-here"
   ```
5. Click **Deploy** — the app will be live in about a minute.

> **Note**: Do **not** commit your `.env` file or `secrets.toml`. Both are listed in `.gitignore`.

---

## Project Structure

```
ResearchPaperSummarizer/
├── app.py                    # Main Streamlit application
├── style.css                 # Custom UI styles
├── requirements.txt          # Python dependencies
├── .env                      # Local secrets (git-ignored)
├── .gitignore
└── .streamlit/
    ├── config.toml           # Streamlit server config
    └── secrets.toml.example  # Template for Streamlit Cloud secrets
```

---

## Model

Uses `gemini-3.5-flash-lite` via the Google Gen AI SDK for fast, cost-efficient inference across all analysis tasks.
