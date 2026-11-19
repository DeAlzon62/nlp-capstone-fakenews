# Fake news detector – NLP capstone

**EDC Paris Business School – Bachelor 3 Data Analytics & AI – Natural Language Processing course**

A web application that estimates whether a news article is **written like a reliable news agency** or **like a website known for publishing false information**, and shows the words behind each decision. Built with the techniques of the course labs (preprocessing, TF-IDF, sentence embeddings, classification) on real data, and tested on recent articles collected by the team.

> ⚠️ The model judges the **writing style** of an article, not whether its facts are true. It is a warning signal, not a fact-checker.

---

## 1. Problem and users

Fake news spreads faster than fact-checkers can verify it. A reader, a student or a community manager often has to decide in seconds whether an article they are about to share looks trustworthy.

**Use case:** paste an English news article → get an indication of its style (reliable agency vs. unreliable partisan site), a confidence score and the words that weighed in the decision, to decide whether it deserves a check with a fact-checker.

## 2. Data

| | Source | Size | Use |
|---|---|---|---|
| Training / evaluation | [GonzaloA/fake_news](https://huggingface.co/datasets/GonzaloA/fake_news) (Hugging Face), built from several Kaggle fake news datasets | 40,587 English articles (title, text, label 1 = real / 0 = fake), official split train 24,353 / validation 8,117 / test 8,117 | notebooks 1-3 |
| Recent test | 30 articles collected by the team on 19/11/2026: 15 from **AP News** (real) and 15 from **The Gateway Pundit** (fake – rated *Questionable*, *Very Low* factual reporting by Media Bias/Fact Check) | 30 articles | notebook 4 |

**What the exploration found (notebook 1):**
- Very clean data: 2 empty texts, no duplicated text, no overlap between splits; 54% real / 46% fake.
- **Strong source bias:** 96% of real articles are Reuters wire stories (`WASHINGTON (Reuters) -`), fake articles come from blogs (titles in capitals 72% vs 2%, *Featured image / image via* 45% vs 0.1%, *VIDEO / WATCH* 28% vs 0%). A model can "cheat" with these clues → they are removed before training.
- Mostly 2015-2017 US politics, English only; labels given **by source**, not article by article.

The data files are **not in this repository** (170 MB > 50 MB limit, and the recent articles are copyrighted). See *How to run* to download them. `recent_articles_sources.csv` lists the titles, URLs and labels of the 30 recent articles.

## 3. Pipeline

```
title + text
   │
   ├─ remove_leaks()      regular expressions: Reuters dateline and mentions, image credits,
   │                      URLs, tweet links, Twitter handles, [Video]/(IMAGE) tags, "WATCH:"
   │
   ├─ A ─ preprocess_advanced()  lowercase, letters only, tokenization, stop words, lemmatization (Lab 2 / 4.2)
   │      └─ TF-IDF (words + bigrams, 50k features) → logistic regression          ← model used by the app
   │
   └─ B ─ title + first 200 words, natural text (no lemmatization)
          └─ sentence embeddings all-MiniLM-L6-v2 (Lab 3.2, 384 dimensions) → logistic regression
```

**Choices:**
- **Title + text** as input: the title carries a lot of style information.
- **Leak removal** before training, and score compared with/without it (97.3% → 96.7% on validation): the model does not depend only on *Reuters*.
- **Logistic regression** rather than linear SVM (97.4% vs 96.7% on validation, close): it gives a probability (confidence score) and readable coefficients (explanation in the app). Naive Bayes: 93.5%.
- **Evaluation:** official test set used only once at the end + comparison TF-IDF vs. embeddings + test on recent articles from other sources.

## 4. Results

| Model | Test set (2016-17, 8,117 articles) | Recent articles (Nov. 2026, 30 articles) |
|---|---|---|
| **A – TF-IDF + logistic regression** | **96.5%** accuracy, F1 macro 96.5% | **83.3%** (real 11/15, fake 14/15) |
| B – sentence embeddings + logistic regression | 92.7% accuracy, F1 macro 92.7% | 60.0% (real 8/15, fake 10/15) |

**Error analysis:**
- **Model A learns the style** (agency vs. blog): *said*, days of the week, *spokesman* → real; *via*, *read*, *watch*, *video*, *like*, *gop* → fake.
- **Label noise:** 4% of the "real" articles of the dataset are not Reuters but lowercase blog texts; they produce 114 of the 281 test errors (61% error rate vs 0.8% for Reuters). They look like probable labelling errors.
- **Model B learns the topic**, which both classes share (US politics): worse everywhere, and on recent articles it calls every non-political real article *fake* and every fake article about Congress *real*.
- **On recent articles**, model A misses soft-news AP stories (sport, health, lifestyle) written with informal words. Same event, two sources (Senator Cornyn blocking a nomination): 0.80 *real* for AP vs 0.35 for The Gateway Pundit.

**Limits:**
- It detects a **style**, not false facts: a false article written in a neutral agency style can pass, a real informal article can be flagged.
- Trained on 2015-2017 English US political news; less reliable on other topics, very short texts, other languages.
- Labels by source (in the dataset and in our recent test); recent test = 30 articles, one source per class, fake set right-wing only (a left-wing questionable site could not be loaded).
- The model can use the name of known sources (*gateway pundit*); an unknown fake news site would not have this clue.

## 5. The application

Streamlit app (`app.py`):
- paste a title and a text (or load one of the two examples),
- verdict: *reliable style* (≥ 65%), *unreliable style* (≤ 35%) or *uncertain*, with the probability,
- chart of the words that pushed the decision towards real or fake,
- "what the model reads": the cleaned text and the number of source clues removed,
- optional comparison with the embedding model,
- **unexpected input handled:** empty text, no words, fewer than 20 words or non-English text → clear message and no verdict; short text, missing title, very long text → warning.

## 6. How to run

**Requirements:** Python 3.11 with the course environment (`uv sync` in the course repository) or `pip install -r requirements.txt`.

1. **Download the data** into `data/`: on the [dataset page](https://huggingface.co/datasets/GonzaloA/fake_news), tab *Files and versions* → folder `data` → the three `.parquet` files (`train-…`, `validation-…`, `test-…`).
2. **Run the notebooks in order** (kernel = course environment, *Restart* then *Run All*):
   - `01_data_exploration.ipynb` – exploration and biases
   - `02_baseline_tfidf_logreg.ipynb` – baseline, evaluation, error analysis, saves `models/baseline_tfidf_logreg.joblib`
   - `03_embeddings_vs_tfidf.ipynb` – comparison with sentence embeddings, saves `models/embeddings_logreg.joblib`
   - `04_recent_articles.ipynb` – test on recent articles (needs `data/recent_articles.csv`, not shared: collect the articles listed in `recent_articles_sources.csv`)

   The two trained models are already in `models/`, so the app works without re-running the notebooks.
3. **Launch the app:**
   ```bash
   # from the course repository (course environment + Streamlit for this run)
   uv run --with streamlit streamlit run ../nlp-capstone-fakenews/app.py
   # or, with pip
   pip install -r requirements.txt
   streamlit run app.py
   ```
   The app opens at http://localhost:8501.

## 7. Repository structure

```
app.py                      Streamlit application
app_examples.json           two example articles from the test set
src/preprocessing.py        leak removal, preprocessing (shared by notebooks and app)
src/model_utils.py          loading, prediction and explanation of the models
notebooks/                  01 exploration – 02 baseline – 03 embeddings – 04 recent articles
models/                     trained models (TF-IDF pipeline 1 MB, embedding classifier 4 kB)
recent_articles_sources.csv titles, URLs and labels of the 30 recent articles (texts not shared)
requirements.txt            Python dependencies
data/                       not versioned (see How to run)
```

## 8. Team

| Member |
|---|---|
| Kilian DALZON |
| Aleksei DAVYDOV |
| Eléonore SION |

AI assistants were used during the project (allowed by the brief); every member can explain and modify each part of the code.
# nlp-capstone-fakenews
