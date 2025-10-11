# 🎈 Blank app template
Automated Resume Shortlisting System

This is a Streamlit application that performs automated resume shortlisting by comparing resumes to a job description using NLP and semantic similarity.

Features
- Upload one job description (TXT/DOCX/PDF) and multiple resumes (PDF/DOCX/TXT)
- Extract text from uploads (PyPDF2, python-docx)
- Extract skills/education heuristically using spaCy
- Compute semantic similarity using Sentence-BERT (recommended) or TF-IDF + cosine similarity as fallback
- Display ranked list of resumes with match percentages
- Show basic analytics (bar chart of top matches, skill frequency)
- Download CSV report of ranked candidates

Requirements
- Python 3.9+
- See `requirements.txt` for versions. Install with:

```bash
pip install -r requirements.txt
```

Important notes
- The app uses spaCy. After installing dependencies, install the English model:

```bash
python -m spacy download en_core_web_sm
```

- Sentence-BERT (sentence-transformers) is optional but recommended for better semantic similarity. If not available, the app falls back to TF-IDF.

Run

```bash
streamlit run streamlit_app.py
```

Usage
- Upload a job description in the sidebar.
- Upload multiple resumes.
- Click "Run shortlisting" to compute scores and view results.

Limitations & next steps
- Skill extraction is heuristic; consider using a curated skill ontology or trained NER for production.
- Add caching for large sets of resumes and background processing for long runs.
- Improve PDF/DOCX parsing for edge cases.

License: MIT
