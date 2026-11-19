"""Fake news detector – Streamlit app (NLP capstone, EDC Paris Business School).

Run locally from the course folder (uses the course environment + Streamlit):
    cd ../nlp-course
    uv run --with streamlit streamlit run ../nlp-capstone-fakenews/app.py
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from src.model_utils import (explain_tfidf, load_baseline, load_embedding_model,  # noqa: E402
                             proba_real_embeddings, proba_real_tfidf)
from src.preprocessing import LEAK_REGEX, build_input, stop_words  # noqa: E402

MIN_WORDS = 20        # below: refused (not enough text to judge a style)
SHORT_WORDS = 80      # below: warning (less reliable, see notebook 2)
MAX_WORDS = 5000      # above: text cut
UNCERTAIN = (0.35, 0.65)

st.set_page_config(page_title="Fake news detector", page_icon="📰", layout="wide")


# ---------------------------------------------------------------- models (loaded once)
@st.cache_resource(show_spinner="Loading the TF-IDF model...")
def get_baseline():
    return load_baseline()


@st.cache_resource(show_spinner="Loading the embedding model (first time: download of ~90 MB)...")
def get_embedding_model():
    try:
        return load_embedding_model()
    except Exception as error:  # model file missing, no internet, sentence-transformers not installed...
        return error


@st.cache_data
def get_examples():
    path = ROOT / "app_examples.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


# ---------------------------------------------------------------- input checks
def check_input(title, text):
    """Return (errors, warnings, text_to_use). Errors block the analysis."""
    errors, warnings = [], []
    text = (text or "").strip()
    title = (title or "").strip()
    words = text.split()

    if not text:
        errors.append("Please paste the text of an article.")
        return errors, warnings, text
    if not re.search(r"[A-Za-z]{2,}", text):
        errors.append("The text contains no words in Latin letters – the model only reads English text.")
        return errors, warnings, text
    if len(words) < MIN_WORDS:
        errors.append(f"The text is too short ({len(words)} words). Paste at least {MIN_WORDS} words: "
                      "the model judges the writing style of an article, which needs more than a sentence.")
        return errors, warnings, text

    tokens = re.findall(r"[a-z]+", text.lower())
    english_share = sum(t in stop_words for t in tokens) / max(len(tokens), 1)
    if english_share < 0.15:
        errors.append("This text does not look like English (very few common English words). "
                      "The model was trained on English articles only, so its answer would not be meaningful.")
        return errors, warnings, text
    if len(words) < SHORT_WORDS:
        warnings.append(f"Short text ({len(words)} words): the result is less reliable than for a full article.")
    if len(words) > MAX_WORDS:
        warnings.append(f"Very long text: only the first {MAX_WORDS} words are analysed.")
        text = " ".join(words[:MAX_WORDS])
    if not title:
        warnings.append("No title given: the title is part of what the model reads, adding it improves the result.")
    return errors, warnings, text


def verdict(p_real):
    if p_real >= UNCERTAIN[1]:
        return "✅ Written like a **reliable news agency**", "green"
    if p_real <= UNCERTAIN[0]:
        return "⚠️ Written like an **unreliable / fake news site**", "red"
    return "❔ **Uncertain** – the style is between the two", "orange"


def contribution_chart(to_real, to_fake):
    words = [w for w, _ in to_fake][::-1] + [w for w, _ in to_real]
    values = [v for _, v in to_fake][::-1] + [v for _, v in to_real]
    fig, ax = plt.subplots(figsize=(6, 0.32 * max(len(words), 4) + 0.8))
    ax.barh(words, values, color=["tab:red" if v < 0 else "tab:blue" for v in values])
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("← pushes towards FAKE        pushes towards REAL →")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    model_choice = st.radio(
        "Model",
        ["TF-IDF + logistic regression (recommended)", "Compare with sentence embeddings"],
        help="The TF-IDF model is more accurate (notebooks 3 and 4). The comparison also runs the "
             "embedding model, which needs sentence-transformers and a download the first time.",
    )
    st.divider()
    st.subheader("How good is it?")
    st.markdown(
        "| | TF-IDF | Embeddings |\n|---|---|---|\n"
        "| Test set (2016-17) | **96.5%** | 92.7% |\n"
        "| Recent articles (Nov. 2026) | **83.3%** | 60.0% |"
    )
    st.caption("Accuracy. Recent articles: 15 AP News vs 15 The Gateway Pundit (notebook 4).")
    st.divider()
    st.subheader("Limits")
    st.markdown(
        "- It judges the **writing style** (news agency vs partisan blog), **not the facts**.\n"
        "- Trained on English US political news from 2015-2017.\n"
        "- Less reliable on soft news (sport, lifestyle), short texts and non-English texts.\n"
        "- Never use it alone to decide if something is true: check with a fact-checker."
    )

# ---------------------------------------------------------------- main page
st.title("📰 Fake news detector")
st.markdown(
    "Paste a news article in English. The app estimates whether it is **written like a reliable news "
    "agency** or **like a website known for publishing false information**, and shows the words "
    "behind the decision."
)

if "title" not in st.session_state:
    st.session_state.title, st.session_state.text = "", ""

examples = get_examples()
if examples:
    cols = st.columns(len(examples) + 1)
    for col, ex in zip(cols, examples):
        if col.button(f"Load: {ex['name']}", use_container_width=True):
            st.session_state.title, st.session_state.text = ex["title"], ex["text"]
    if cols[-1].button("Clear", use_container_width=True):
        st.session_state.title, st.session_state.text = "", ""

title = st.text_input("Title", key="title", placeholder="Headline of the article")
text = st.text_area("Text", key="text", height=260, placeholder="Paste the body of the article here")
analyse = st.button("Analyse", type="primary")

if analyse:
    errors, warnings, text_used = check_input(title, text)
    for message in errors:
        st.error(message)
    if not errors:
        for message in warnings:
            st.warning(message)

        baseline = get_baseline()
        p_real = float(proba_real_tfidf(baseline, [title], [text_used])[0])
        label, colour = verdict(p_real)

        st.subheader("Result")
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"### :{colour}[{label}]", unsafe_allow_html=False)
            st.metric("Probability 'reliable style' (TF-IDF model)", f"{p_real:.0%}")
            st.progress(p_real)
            st.caption("Above 65%: reliable style – below 35%: unreliable style – in between: uncertain.")

            if model_choice.startswith("Compare"):
                emb = get_embedding_model()
                if isinstance(emb, Exception):
                    st.info(f"The embedding model is not available here ({type(emb).__name__}). "
                            "Only the TF-IDF result is shown.")
                else:
                    encoder, emb_clf, max_words = emb
                    p_emb = float(proba_real_embeddings(encoder, emb_clf, [title], [text_used], max_words)[0])
                    st.metric("Probability 'reliable style' (embedding model)", f"{p_emb:.0%}",
                              delta=f"{(p_emb - p_real) * 100:+.0f} points vs TF-IDF", delta_color="off")
                    st.caption("The embedding model judges the topic more than the style: "
                               "it was less accurate in our tests.")

        with right:
            to_real, to_fake = explain_tfidf(baseline, title, text_used, top_n=8)
            st.markdown("**Words that weighed the most in the decision**")
            if to_real or to_fake:
                st.pyplot(contribution_chart(to_real, to_fake))
            else:
                st.info("None of the words of this text is in the model's vocabulary.")

        with st.expander("What the model reads (after cleaning)"):
            removed = sum(len(r.findall(s)) for r in LEAK_REGEX for s in (title, text_used))
            st.markdown(f"Source clues removed before the analysis (Reuters dateline, image credits, "
                        f"URLs, Twitter handles, video tags): **{removed}**")
            st.code(build_input(title, text_used)[:1500] or "(empty)", language=None)

st.divider()
st.caption("NLP capstone project – EDC Paris Business School. Dataset: GonzaloA/fake_news (Hugging Face). "
           "This tool estimates a writing style; it does not verify facts.")
