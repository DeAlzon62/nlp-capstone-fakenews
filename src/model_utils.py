"""Helpers shared by notebook 4 and the Streamlit app: load the models, predict, explain."""
from pathlib import Path

import joblib
import numpy as np

from src.preprocessing import build_input, embedding_input

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_baseline():
    """TF-IDF + logistic regression pipeline saved by notebook 2."""
    return joblib.load(MODEL_DIR / "baseline_tfidf_logreg.joblib")


def load_embedding_model():
    """(encoder, classifier, max_words) saved by notebook 3. Downloads all-MiniLM-L6-v2 the first time."""
    from sentence_transformers import SentenceTransformer
    saved = joblib.load(MODEL_DIR / "embeddings_logreg.joblib")
    return SentenceTransformer(saved["encoder_name"]), saved["classifier"], saved["max_words"]


def proba_real_tfidf(model, titles, texts):
    """Probability of 'real' (label 1) for each article, TF-IDF model."""
    return model.predict_proba([build_input(t, x) for t, x in zip(titles, texts)])[:, 1]


def proba_real_embeddings(encoder, classifier, titles, texts, max_words=200):
    """Probability of 'real' (label 1) for each article, embedding model."""
    inputs = [embedding_input(t, x, max_words) for t, x in zip(titles, texts)]
    emb = encoder.encode(inputs, batch_size=32, normalize_embeddings=True)
    return classifier.predict_proba(emb)[:, 1]


def explain_tfidf(model, title, text, top_n=8):
    """Words of ONE article that pushed the TF-IDF model the most.

    contribution of a word = its TF-IDF value in the article x its coefficient.
    Positive -> pushes towards real, negative -> pushes towards fake.
    Returns two lists of (word, contribution): towards real, towards fake.
    """
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    x = tfidf.transform([build_input(title, text)])
    vocab = tfidf.get_feature_names_out()
    idx = x.nonzero()[1]
    contrib = x.toarray()[0, idx] * clf.coef_[0, idx]
    order = np.argsort(contrib)
    to_fake = [(vocab[idx[i]], float(contrib[i])) for i in order[:top_n] if contrib[i] < 0]
    to_real = [(vocab[idx[i]], float(contrib[i])) for i in order[::-1][:top_n] if contrib[i] > 0]
    return to_real, to_fake
