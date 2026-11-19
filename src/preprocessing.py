"""Text preprocessing shared by the notebooks and the Streamlit app.

The functions preprocess_simple and preprocess_advanced are the ones of Lab 4.2.
remove_leaks removes the "source clues" found in notebook 1: they tell where an
article comes from (Reuters, a blog...), not whether its content is true.
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(resource, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Source clues found in notebook 1 (regular expressions, Lab 1)
LEAK_PATTERNS = [
    r"^\s*[A-Z][A-Za-z .,/'-]{1,60}\(Reuters\)\s*-\s*",  # dateline "WASHINGTON (Reuters) - "
    r"\breuters\b",                                      # any other mention of Reuters
    r"featured image[^.\n]*",                            # "Featured image via Getty Images"
    r"\bimage via\b[^.\n]*",                             # "image via Twitter"
    r"https?://\S+|www\.\S+",                            # URLs
    r"\b(?:pic\.)?twitter\.com/\S+",                     # embedded tweets "pic.twitter.com/abc"
    r"@\w+",                                             # Twitter handles
    r"[\[(]\s*(?:video|image|images|tweet|tweets|watch)\s*[\])]",  # "[Video]", "(IMAGE)"
    r"^\s*(?:watch|breaking)\s*:?",                      # title starting with "WATCH:"
]
LEAK_REGEX = [re.compile(p, flags=re.IGNORECASE | re.MULTILINE) for p in LEAK_PATTERNS]


def remove_leaks(text):
    """Remove the source clues (Reuters dateline, image credits, URLs, handles, video tags)."""
    text = str(text)
    for regex in LEAK_REGEX:
        text = regex.sub(" ", text)
    return text


def preprocess_simple(text):
    """Lowercase, letters only, single spaces (Lab 4.2)."""
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    return " ".join(text.split())


def preprocess_advanced(text):
    """preprocess_simple + stop word removal + lemmatization + words of 3+ characters (Lab 4.2)."""
    tokens = word_tokenize(preprocess_simple(text))
    return " ".join(lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2)


def embedding_input(title, text, max_words=200):
    """Input of the sentence-embedding model (notebook 3): leaks removed, natural text kept,
    title + first `max_words` words (all-MiniLM-L6-v2 reads at most 256 tokens)."""
    title, text = remove_leaks(title), remove_leaks(text)
    words = " ".join(str(text).split()[:max_words])
    return f"{' '.join(str(title).split())}. {words}"


def build_input(title, text, clean_leaks=True):
    """Title + body of an article, ready for the TF-IDF vectorizer."""
    if clean_leaks:  # title and text cleaned separately: some patterns only match at the start
        title, text = remove_leaks(title), remove_leaks(text)
    return preprocess_advanced(f"{title}. {text}")
