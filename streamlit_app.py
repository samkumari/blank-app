import io
import sys
import subprocess
from collections import Counter
from typing import List, Dict
from difflib import SequenceMatcher

import streamlit as st
import pandas as pd
import numpy as np

# Dynamic imports with graceful error messages
missing_packages = []

def try_import(name: str, import_name: str = None, pip_name: str = None):
    import_name = import_name or name
    pip_name = pip_name or name
    try:
        module = __import__(import_name)
        return module
    except Exception:
        missing_packages.append(pip_name)
        return None

# Core libraries
pd = try_import('pandas')
np = try_import('numpy')

# PDF and docx
PdfReader = None
docx = None
try:
    from PyPDF2 import PdfReader as _PdfReader
    PdfReader = _PdfReader
except Exception:
    missing_packages.append('PyPDF2')

try:
    import docx as _docx
    docx = _docx
except Exception:
    missing_packages.append('python-docx')

# sklearn
TfidfVectorizer = None
cosine_similarity = None
try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    TfidfVectorizer = _TfidfVectorizer
    cosine_similarity = _cosine_similarity
except Exception:
    missing_packages.append('scikit-learn')

# sentence-transformers (optional)
SBERT_AVAILABLE = False
SentenceTransformer = None
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    SentenceTransformer = _SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    missing_packages.append('sentence-transformers')

# spaCy
spacy = None
try:
    import spacy as _spacy
    spacy = _spacy
except Exception:
    missing_packages.append('spacy')


st.set_page_config(page_title="Automated Resume Shortlisting System", layout="wide")

# Basic CSS
st.markdown(
    """
    <style>
    .app-header {display:flex; align-items:center; gap:16px}
    .candidate-card {border:1px solid rgba(255,255,255,0.06); padding:12px; border-radius:8px; margin-bottom:8px}
    .metric {background:#0e1117; padding:12px; border-radius:8px; text-align:center}
    .skill-pill {display:inline-block; padding:4px 8px; margin:2px; background:#262935; border-radius:999px; color:#fff; font-size:12px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-header"><h1>Automated Resume Shortlisting System</h1></div>', unsafe_allow_html=True)

# Handle missing packages
if missing_packages:
    st.error("The following packages are missing: " + ', '.join(sorted(set(missing_packages))))
    st.stop()

# ============================ CURATED SKILL LIST ============================
CURATED_SKILLS = {
    'python', 'java', 'javascript', 'react', 'reactjs', 'html', 'css', 'sql', 'machine learning',
    'deep learning', 'nlp', 'data analysis', 'data visualization', 'django', 'flask',
    'tensorflow', 'pytorch', 'power bi', 'excel', 'communication', 'leadership',
    'problem solving', 'c++', 'git', 'docker', 'kubernetes', 'aws', 'linux',
    'time management', 'project management', 'mongodb', 'node.js', 'typescript',
    'streamlit', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'data science', 'ai',
    'html5', 'css3', 'bootstrap', 'json', 'api', 'sql server', 'jupyter', 'mlops', 'azure'
}

# ============================ TEXT EXTRACTION ============================
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode(errors="ignore")

def extract_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    content = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(content)
    if name.endswith(".docx"):
        return extract_text_from_docx(content)
    if name.endswith(".txt"):
        return extract_text_from_txt(content)
    return ""

# ============================ NLP UTILITIES ============================
@st.cache_data(show_spinner=False)
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None

nlp = load_spacy_model()

def normalize_phrase(s: str) -> str:
    return ' '.join(s.lower().strip().split())

def fuzzy_match(skill1: str, skill2: str, threshold=0.8) -> bool:
    return SequenceMatcher(None, skill1, skill2).ratio() >= threshold

# Enhanced skill extraction
def get_skill_set(text: str, nlp_model) -> set:
    if not text:
        return set()

    text_lower = text.lower()
    tokens = set(word.strip(".,:-_()[]") for word in text_lower.split())
    extracted_skills = set()

    # Curated skill match
    for skill in CURATED_SKILLS:
        if skill in text_lower:
            extracted_skills.add(skill)

    # Heuristic: noun chunks
    if nlp_model:
        doc = nlp_model(text)
        for chunk in doc.noun_chunks:
            phrase = chunk.text.lower().strip()
            if len(phrase.split()) <= 4:
                extracted_skills.add(phrase)

    # Fuzzy match expansion
    for token in tokens:
        for skill in CURATED_SKILLS:
            if fuzzy_match(token, skill, threshold=0.85):
                extracted_skills.add(skill)

    return set(normalize_phrase(s) for s in extracted_skills if len(s) > 2)

# ============================ SIMILARITY CALCULATION ============================
@st.cache_data(show_spinner=False)
def compute_sentence_embeddings(texts: List[str]):
    if SBERT_AVAILABLE:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(texts, show_progress_bar=False)
    else:
        return None

def compute_tfidf_similarity(job_text: str, resumes: List[str]) -> List[float]:
    corpus = [job_text] + resumes
    vec = TfidfVectorizer(stop_words='english')
    mat = vec.fit_transform(corpus)
    sims = cosine_similarity(mat[0:1], mat[1:])[0]
    return list(sims)

def compute_sbert_similarity(job_text: str, resumes: List[str]) -> List[float]:
    texts = [job_text] + resumes
    emb = compute_sentence_embeddings(texts)
    if emb is None:
        return compute_tfidf_similarity(job_text, resumes)
    sims = cosine_similarity([emb[0]], emb[1:])[0]
    return list(sims)

def prepare_report(rows: List[dict]) -> str:
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

# ============================ STREAMLIT UI ============================
with st.sidebar:
    st.header("Upload files")
    job_file = st.file_uploader("Upload Job Description", type=["pdf", "docx", "txt"])
    resume_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    use_sbert = st.checkbox("Use Sentence-BERT (recommended)", value=True)
    run_btn = st.button("Run Shortlisting")

st.info("Supported formats: PDF, DOCX, TXT. For best results, use clear text resumes.")

if not nlp:
    st.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")

# ============================ MAIN LOGIC ============================
if run_btn:
    job_text = extract_text(job_file)
    if not job_text.strip():
        st.error("Please upload a valid job description file.")
    elif not resume_files:
        st.error("Please upload at least one resume file.")
    else:
        with st.spinner("Analyzing resumes..."):
            job_skills = get_skill_set(job_text, nlp)
            resumes_texts = [extract_text(f) for f in resume_files]
            filenames = [f.name for f in resume_files]

            # Similarity calculation
            sims = compute_sbert_similarity(job_text, resumes_texts) if use_sbert and SBERT_AVAILABLE else compute_tfidf_similarity(job_text, resumes_texts)

            rows = []
            for name, score, text in zip(filenames, sims, resumes_texts):
                resume_skills = get_skill_set(text, nlp)

                # Improved fuzzy matching
                matched_skills = []
                for js in job_skills:
                    for rs in resume_skills:
                        if fuzzy_match(js, rs, threshold=0.82):
                            matched_skills.append(js)
                matched_skills = sorted(list(set(matched_skills)))

                rows.append({
                    "filename": name,
                    "match_percent": float(score) * 100,
                    "matched_skills_count": len(matched_skills),
                    "matched_skills": ", ".join(matched_skills),
                    "total_skills": len(resume_skills),
                })

            rows_sorted = sorted(rows, key=lambda r: r['match_percent'], reverse=True)
            df = pd.DataFrame(rows_sorted)

            st.subheader("Ranked Candidates")
            df_display = df.copy()
            df_display['match_percent'] = df_display['match_percent'].map(lambda x: f"{x:.1f}%")
            st.dataframe(df_display, width='stretch')

            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric('Top Score', f"{df.iloc[0]['match_percent']:.1f}%")
            col2.metric('Candidates', len(df))
            col3.metric('Average Match', f"{df['match_percent'].mean():.1f}%")

            st.bar_chart(df.set_index('filename')['match_percent'])

            csv_bytes = prepare_report(rows_sorted).encode('utf-8')
            st.download_button("Download CSV Report", data=csv_bytes, file_name="shortlisting_report.csv", mime="text/csv")

            st.success("✅ Skill matching improved — fuzzy & curated-based detection complete!")
