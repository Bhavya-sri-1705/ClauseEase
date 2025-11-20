# File: app.py

import streamlit as st
import pandas as pd
import nltk
import spacy
import fitz  # PyMuPDF
import textstat
import re
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Transformers imports guarded (if installed)
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# ---------------------------
# BASIC CONFIG / SETUP
# ---------------------------
st.set_page_config(page_title="Clause Ease — Text Simplifier", layout="wide")

# Download minimal NLTK data (idempotent)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load spaCy model (download if missing)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download as spacy_download
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Ensure folders
Path("logs").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# ---------------------------
# STYLES (kept from your original)
# ---------------------------
st.markdown("""
    <style>
        /* animated background + styling (kept minimal) */
        @keyframes gradientMove {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #a2d2ff, #b9fbc0, #ffe6a7, #ffd6e0);
            background-size: 400% 400%;
            animation: gradientMove 18s ease infinite;
            color: #222;
        }
        [data-testid="stSidebar"] { color: white !important; }
        div.stButton > button { border-radius: 10px; padding: 0.6em 1.2em; font-weight: bold; }
        input, textarea, select { border-radius: 8px !important; border: 1px solid #90caf9 !important; }
        h1, h2, h3, h4, h5 { font-family: 'Poppins', sans-serif; color: #0d47a1; }
        .card { background: rgba(255,255,255,0.9); border-radius: 15px; padding: 20px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# GLOSSARY (single source)
# ---------------------------
GLOSSARY_PATH = "glossary.json"
default_glossary = {
    "plasmid": "A small DNA molecule within a cell that is physically separated from chromosomal DNA.",
    "mitosis": "A process where a single cell divides into two identical daughter cells.",
    "photosynthesis": "Process by which plants convert sunlight into chemical energy."
}
if not os.path.exists(GLOSSARY_PATH):
    with open(GLOSSARY_PATH, "w", encoding="utf-8") as f:
        json.dump(default_glossary, f, indent=2)

with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
    GLOSSARY: Dict[str, str] = json.load(f)

# ---------------------------
# HELPERS
# ---------------------------
def clean_text(text: str, lower: bool = True, remove_extra_whitespace: bool = True) -> str:
    text = text.replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[“”‘’]', '"', text)
    if lower:
        text = text.lower()
    if remove_extra_whitespace:
        text = ' '.join(text.split())
    return text.strip()

def extract_text_from_pdf(pdf_bytes) -> str:
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")
    return text

def sentence_segment(text: str):
    if nlp is not None:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    else:
        return nltk.sent_tokenize(text)

def tokenize(text: str):
    if nlp is not None:
        doc = nlp(text)
        return [token.text for token in doc if not token.is_space]
    else:
        return nltk.word_tokenize(text)

def pos_tags(text: str):
    if nlp is not None:
        doc = nlp(text)
        return [(token.text, token.pos_) for token in doc]
    else:
        return nltk.pos_tag(tokenize(text))

def compute_readability(text: str):
    try:
        scores = {
            "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(text),
            "Gunning Fog Index": textstat.gunning_fog(text),
            "SMOG Index": textstat.smog_index(text),
            "Automated Readability Index": textstat.automated_readability_index(text)
        }
    except Exception:
        # If textstat fails for empty input or other reasons, return zeros
        scores = { "Flesch-Kincaid Grade": 0, "Gunning Fog Index": 0, "SMOG Index": 0, "Automated Readability Index": 0 }
    return scores

# ---------------------------
# LOGIN (file-based simple system)
# ---------------------------
USERS_FILE = "users.json"
if not os.path.exists(USERS_FILE):
    initial_users = {"admin": "admin123", "user": "user123"}
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(initial_users, f, indent=2)

def load_users():
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(data):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "register_mode" not in st.session_state:
    st.session_state.register_mode = False

def login_page():
    st.title("🔐 Welcome to ClauseEase AI")
    st.write("Please log in to access the dashboard or register for a new account.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load_users()
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome back, {username}!")
            time.sleep(1)
            st.rerun()

        else:
            st.error("Invalid username or password. Please try again or register.")

    if st.button("Register New User"):
        st.session_state.register_mode = True
        st.rerun()

def register_page():
    st.title("📝 Register New Account")
    new_user = st.text_input("Choose a Username")
    new_pass = st.text_input("Choose a Password", type="password")
    confirm_pass = st.text_input("Confirm Password", type="password")

    if st.button("⬅ Back to Login"):
        st.session_state.register_mode = False
        st.rerun()

    if st.button("✅ Register Account"):
        users = load_users()
        if new_user in users:
            st.error("Username already exists. Please choose another.")
        elif not new_user or not new_pass:
            st.warning("Username and password cannot be empty.")
        elif new_pass != confirm_pass:
            st.error("Passwords do not match.")
        else:
            users[new_user] = new_pass
            save_users(users)
            st.success(f"Account created successfully for {new_user}! Please log in.")
            time.sleep(1)
            st.session_state.register_mode = False
            st.rerun()

def logout_button():
    st.sidebar.write(f"👤 Logged in as: *{st.session_state.username}*")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

# ---------------------------
# LOGGING
# ---------------------------
LOG_PATH = "logs/simplification_requests.json"
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)

def log_request(entry: Dict[str, Any]):
    with open(LOG_PATH, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data.append(entry)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

# ---------------------------
# MODELS / SIMPLIFICATION
# ---------------------------
# Small set of models (adjust if you want different models)
SIMPLIFICATION_MODELS = {
    "Basic": ("google/flan-t5-small", {"max_length": 128, "temperature": 0.2, "num_beams": 4}),
    "Intermediate": ("google/flan-t5-base", {"max_length": 200, "temperature": 0.6, "num_beams": 4}),
    "Advanced": ("philschmid/bart-large-cnn-samsum", {"max_length": 256, "temperature": 0.9, "num_beams": 3})
}

def naive_simplify(text: str, ratio: float = 0.5) -> str:
    sentences = sentence_segment(text)
    keep = max(1, int(len(sentences) * (1 - ratio)))
    simple = " ".join(sentences[:keep])
    replacements = {
        "utilize": "use",
        "commence": "start",
        "terminate": "end",
        "demonstrate": "show",
    }
    for k, v in replacements.items():
        simple = re.sub(rf"\b{k}\b", v, simple, flags=re.IGNORECASE)
    return simple

# Create and cache pipelines only if transformers installed
if TRANSFORMERS_AVAILABLE:
    @st.cache_resource(ttl=60*60*12)
    def load_hf_model(model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
        return tokenizer, model, pipe
else:
    def load_hf_model(model_name: str):
        raise RuntimeError("Transformers not installed in the environment.")

def simplify_text_with_level(text: str, level: str):
    model_info = SIMPLIFICATION_MODELS.get(level)
    if not model_info:
        return naive_simplify(text, ratio=0.5), "fallback"

    model_name, gen_params = model_info
    if not TRANSFORMERS_AVAILABLE:
        return naive_simplify(text, ratio=0.5), "fallback"

    try:
        tokenizer, model, pipe = load_hf_model(model_name)
        prompt = f"Simplify the following text for a general audience:\n\n{text}"
        out = pipe(prompt,
                   max_length=gen_params.get("max_length", 150),
                   num_return_sequences=1,
                   num_beams=gen_params.get("num_beams", 4),
                   temperature=gen_params.get("temperature", 0.2))
        # pipeline output may vary by version; try common keys
        if isinstance(out, list) and len(out) > 0:
            generated = out[0].get("generated_text") or out[0].get("text") or out[0].get("summary_text") or ""
        else:
            generated = ""
        return generated, model_name
    except Exception as e:
        print("Model load/generation failed:", e)
        return naive_simplify(text, ratio=0.5), "fallback"

# Summarization (similar approach)
SUMMARIZATION_MODELS = {
    "default": ("facebook/bart-large-cnn", {"max_length": 130, "min_length": 30, "num_beams": 4})
}

if TRANSFORMERS_AVAILABLE:
    @st.cache_resource(ttl=60*60*12)
    def load_summarizer(model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
        return pipe
else:
    def load_summarizer(model_name: str):
        raise RuntimeError("Transformers not installed in the environment.")

def summarize_text(text: str, model_key="default"):
    model_name, gen_params = SUMMARIZATION_MODELS[model_key]
    if not TRANSFORMERS_AVAILABLE:
        sents = sentence_segment(text)
        return " ".join(sents[:3]), "fallback"
    try:
        pipe = load_summarizer(model_name)
        summary = pipe(text, max_length=gen_params["max_length"], min_length=gen_params["min_length"], do_sample=False)[0].get("summary_text", "")
        return summary, model_name
    except Exception as e:
        print("Summarizer failed:", e)
        sents = sentence_segment(text)
        return " ".join(sents[:3]), "fallback"

# ---------------------------
# HIGHLIGHTING TERMS (glossary)
# ---------------------------
def highlight_terms_html(text: str, glossary: Dict[str, str]):
    def replace_term(match):
        term = match.group(0)
        explanation = glossary.get(term.lower(), glossary.get(term, ""))
        safe_expl = explanation.replace('"', "&quot;")
        return f'<span style="background:#fff7cc; border-radius:3px;" title="{safe_expl}">{term}</span>'
    html = text
    for term in sorted(glossary.keys(), key=lambda s: -len(s)):
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', flags=re.IGNORECASE)
        html = pattern.sub(replace_term, html)
    return "<div>" + html.replace("\n", "<br/>") + "</div>"

# ---------------------------
# DASHBOARD
# ---------------------------
def main_dashboard():
    st.markdown("""
        <h2 style='color:white; background:linear-gradient(90deg,#003366,#3366cc); padding:12px; border-radius:10px;'>
            🧠 Text Preprocessing, Simplification & Readability Analysis
        </h2>
    """, unsafe_allow_html=True)
    st.write("Document Ingestion(upload PDF,TXT,DOCX).")

    logout_button()

    st.sidebar.header("⚙ Options")
    lower = st.sidebar.checkbox("Convert text to lowercase", True)
    show_pos = st.sidebar.checkbox("Show Part-of-Speech Tags", False)
    show_tokens = st.sidebar.checkbox("Show Tokens", False)
    show_sentences = st.sidebar.checkbox("Show Sentences", False)

    st.sidebar.markdown("### 🧾 Simplification & Summarization")
    simpl_level = st.sidebar.selectbox("Simplification Level", ["Basic", "Intermediate", "Advanced"])
    enable_simpl = st.sidebar.checkbox("Enable Simplification", value=True)
    enable_summary = st.sidebar.checkbox("Generate Summary", value=True)
    side_by_side = st.sidebar.checkbox("Show Side-by-Side (Original vs Simplified)", value=True)
    highlight_terms = st.sidebar.checkbox("Highlight Key Terms (Glossary)", value=True)

    uploaded_file = st.file_uploader("📤 Upload a .txt or .pdf file", type=["txt", "pdf"])
    user_text = ""

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                user_text = extract_text_from_pdf(uploaded_file.read())
            else:
                user_text = uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error("Failed to read uploaded file.")
            user_text = ""
    else:
        user_text = st.text_area("Or paste your text below:", height=250)

    if st.sidebar.button("Analyze"):
        if not user_text or not user_text.strip():
            st.warning("Please provide text or upload a file before analysis.")
        else:
            cleaned_text = clean_text(user_text, lower)
            sentences = sentence_segment(cleaned_text)
            tokens = tokenize(cleaned_text)
            scores = compute_readability(cleaned_text)

            # Readability metrics
            st.subheader("📊 Readability Metrics")
            col1, col2, col3, col4 = st.columns(4)
            cols = [col1, col2, col3, col4]
            for i, (metric, score) in enumerate(scores.items()):
                with cols[i]:
                    st.metric(metric, round(score, 2))

            st.markdown("---")
            st.subheader("📈 Basic Statistics")
            st.write(f"*Total Sentences:* {len(sentences)}")
            st.write(f"*Total Words:* {len(tokens)}")
            st.write(f"*Total Characters:* {len(cleaned_text)}")

            if show_sentences:
                st.subheader("✂ Sentences")
                st.write(pd.DataFrame(sentences, columns=["Sentence"]))

            if show_tokens:
                st.subheader("🔤 Tokens")
                st.write(pd.DataFrame(tokens, columns=["Token"]))

            if show_pos:
                st.subheader("🧩 POS Tags")
                st.write(pd.DataFrame(pos_tags(cleaned_text), columns=["Token", "POS"]))

            # Summarization
            summary_text = None
            summary_model = None
            if enable_summary:
                with st.spinner("Generating summary..."):
                    summary_text, summary_model = summarize_text(cleaned_text)
                st.subheader("📝 Summary")
                st.write(summary_text)
                st.caption(f"Model: {summary_model}")

            # Simplification
            simplified_text = None
            used_model = None
            if enable_simpl:
                with st.spinner(f"Simplifying text (level={simpl_level})..."):
                    simplified_text, used_model = simplify_text_with_level(cleaned_text, simpl_level)
                st.subheader("🛠 Simplified Text")
                if side_by_side:
                    left, right = st.columns(2)
                    with left:
                        st.markdown("*Original (complex)*")
                        if highlight_terms:
                            st.markdown(highlight_terms_html(cleaned_text, GLOSSARY), unsafe_allow_html=True)
                        else:
                            st.write(cleaned_text)
                    with right:
                        st.markdown(f"*Simplified — Level: {simpl_level}*")
                        st.write(simplified_text)
                else:
                    if highlight_terms:
                        st.markdown(highlight_terms_html(simplified_text, GLOSSARY), unsafe_allow_html=True)
                    else:
                        st.write(simplified_text)
                st.caption(f"Simplification model used: {used_model}")

                # Log request
                entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "user": st.session_state.username,
                    "level": simpl_level,
                    "model_used": used_model,
                    "summary_used": summary_model if enable_summary else None,
                    "input_snippet": cleaned_text[:500],
                    "output_snippet": (simplified_text or "")[:500]
                }
                log_request(entry)

            st.download_button("📥 Download Cleaned Text", cleaned_text, file_name="cleaned_text.txt")
            if simplified_text:
                st.download_button("📥 Download Simplified Text", simplified_text, file_name="simplified_text.txt")
            if summary_text:
                st.download_button("📥 Download Summary", summary_text, file_name="summary.txt")

# ---------------------------
# APP CONTROLLER
# ---------------------------
if st.session_state.logged_in:
    main_dashboard()
elif st.session_state.register_mode:
    register_page()
else:
    login_page()
