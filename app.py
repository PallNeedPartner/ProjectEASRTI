import re
import os
import sys
import numpy as np
import joblib
import sqlite3

from flask import Flask, render_template, request, redirect

app = Flask(__name__)


# ── Auto-download model kalau belum ada ──────────────────────────
REQUIRED_FILES = [
    "model/fake_news_model.pkl",
    "model/tfidf_vectorizer.pkl",
    "model/nlm_classifier.pkl",
    "model/word2vec.pkl",
]
if not all(os.path.exists(f) for f in REQUIRED_FILES):
    print("⬇️  Model tidak ditemukan, downloading dari Google Drive...")
    import download_models


# ── Load kedua model ──────────────────────────────────────────────
# Model 1: TF-IDF + Logistic Regression (NLP klasik)
tfidf_model      = joblib.load("model/fake_news_model.pkl")
tfidf_vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Model 2: Word2Vec + Logistic Regression (NLM)
nlm_classifier   = joblib.load("model/nlm_classifier.pkl")
w2v_model        = joblib.load("model/word2vec.pkl")


# ── Helper: NLM tokenize & embed ─────────────────────────────────
def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

def doc_to_vec(tokens, model, vector_size=100):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


# ── DB helper ─────────────────────────────────────────────────────
def get_db():
    return sqlite3.connect("database/news.db")


# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    text       = request.form.get("news", "").strip()
    model_mode = request.form.get("model_mode", "tfidf")  # "tfidf" atau "nlm"

    if not text:
        return render_template(
            "index.html",
            result="Input berita tidak boleh kosong.",
            confidence=0,
            model_used="-"
        )

    # ── Prediksi sesuai model yang dipilih user ───────────────────
    if model_mode == "nlm":
        tokens     = simple_tokenize(text)
        vec        = doc_to_vec(tokens, w2v_model).reshape(1, -1)
        prediction = nlm_classifier.predict(vec)[0]
        confidence = max(nlm_classifier.predict_proba(vec)[0])
        model_used = "NLM (Word2Vec)"
    else:
        vec        = tfidf_vectorizer.transform([text])
        prediction = tfidf_model.predict(vec)[0]
        confidence = max(tfidf_model.predict_proba(vec)[0])
        model_used = "NLP (TF-IDF)"

    result = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

    # ── Simpan ke history ─────────────────────────────────────────
    conn = get_db()
    conn.execute(
        """
        INSERT INTO history (news_text, prediction, confidence, model_used)
        VALUES (?, ?, ?, ?)
        """,
        (text, result, float(confidence), model_used)
    )
    conn.commit()
    conn.close()

    return render_template(
        "index.html",
        result=result,
        confidence=round(confidence * 100, 2),
        model_used=model_used
    )


@app.route("/history")
def history():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM history ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return render_template("history.html", rows=rows)


@app.route("/clear-history")
def clear_history():
    conn = get_db()
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return redirect("/history")


@app.route("/dashboard")
def dashboard():
    conn = get_db()

    fake = conn.execute(
        "SELECT COUNT(*) FROM history WHERE prediction='FAKE NEWS'"
    ).fetchone()[0]

    real = conn.execute(
        "SELECT COUNT(*) FROM history WHERE prediction='REAL NEWS'"
    ).fetchone()[0]

    tfidf_count = conn.execute(
        "SELECT COUNT(*) FROM history WHERE model_used='NLP (TF-IDF)'"
    ).fetchone()[0]

    nlm_count = conn.execute(
        "SELECT COUNT(*) FROM history WHERE model_used='NLM (Word2Vec)'"
    ).fetchone()[0]

    conn.close()

    return render_template(
        "dashboard.html",
        fake=fake,
        real=real,
        tfidf_count=tfidf_count,
        nlm_count=nlm_count
    )

@app.route("/delete/<int:id>")
def delete_history(id):
    conn = get_db()
    conn.execute("DELETE FROM history WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    return redirect("/history")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)