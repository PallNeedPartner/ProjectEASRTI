"""
train_nlm.py — Word2Vec Neural Language Model
=============================================
Model kedua untuk FakeShield AI.
Cara kerja:
  1. Word2Vec → setiap kata jadi vector 100 dimensi
  2. Setiap artikel → rata-rata semua vector katanya (document embedding)
  3. Hasil embedding → masuk ke Logistic Regression
Keunggulan vs TF-IDF:
  - Nangkep makna semantik kata (sinonim, konteks)
  - Tidak bergantung pada nama sumber berita (Reuters dll)
  - Lebih general untuk teks di luar dataset training
"""

import re
import numpy as np
import pandas as pd
import joblib

from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ── 1. Load dataset ───────────────────────────────────────────────
print("📂 Loading dataset...")

fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Gabungkan title + text, fillna biar aman
data["text"] = data["title"].fillna("") + " " + data["text"].fillna("")

# Strip prefix sumber berita "(Reuters) —" dll
data["text"] = data["text"].apply(
    lambda x: re.sub(r"^.*?\(.*?\)\s*[-–]?\s*", "", x)
)


# ── 2. Tokenisasi ─────────────────────────────────────────────────
print("✂️  Tokenizing...")

def simple_tokenize(text):
    """Lowercase, buang karakter non-huruf, split per kata."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

data["tokens"] = data["text"].apply(simple_tokenize)

# Buang baris yang tokennya kosong
data = data[data["tokens"].map(len) > 0].reset_index(drop=True)


# ── 3. Train Word2Vec ─────────────────────────────────────────────
print("🧠 Training Word2Vec...")

w2v_model = Word2Vec(
    sentences=data["tokens"].tolist(),
    vector_size=100,       # dimensi tiap vector kata
    window=5,              # konteks 5 kata kiri-kanan
    min_count=2,           # abaikan kata yang muncul < 2x
    workers=4,
    epochs=10,
    sg=1                   # Skip-Gram (lebih baik untuk rare words)
)

print(f"   Vocabulary size: {len(w2v_model.wv):,} kata")


# ── 4. Document Embedding ─────────────────────────────────────────
# Rata-ratakan semua vector kata dalam satu artikel
# → menghasilkan 1 vector 100-dimensi per artikel

def doc_to_vec(tokens, model, vector_size=100):
    vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

print("📐 Creating document embeddings...")
X = np.array([
    doc_to_vec(tokens, w2v_model)
    for tokens in data["tokens"]
])
y = data["label"].values


# ── 5. Train Classifier ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("🏋️  Training Logistic Regression on embeddings...")

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=1.0
)
clf.fit(X_train, y_train)


# ── 6. Evaluasi ───────────────────────────────────────────────────
pred = clf.predict(X_test)
acc  = accuracy_score(y_test, pred)

print("\n" + "=" * 50)
print(f"NLM Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print("=" * 50)
print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=["FAKE NEWS", "REAL NEWS"]))


# ── 7. Simpan ─────────────────────────────────────────────────────
joblib.dump(clf,      "model/nlm_classifier.pkl")
joblib.dump(w2v_model,"model/word2vec.pkl")

print("\n✅ NLM berhasil disimpan!")
print("   → model/nlm_classifier.pkl")
print("   → model/word2vec.pkl")