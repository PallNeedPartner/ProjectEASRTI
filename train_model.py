import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# ── 1. Load dataset ──────────────────────────────────────────────
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real], ignore_index=True)

# ── 2. Fix: gabungkan kolom text dengan fillna ────────────────────
# Tanpa ini, baris yang kolom title/text-nya NaN → crash saat training
data["text"] = data["title"].fillna("") + " " + data["text"].fillna("")

# ── 3. Fix: buang sumber berita dari teks ────────────────────────
# Dataset Kaggle ini sering mengandung prefix "(Reuters)" atau
# nama sumber lain di awal teks. Model jadi belajar NAMA SUMBER
# bukan ISI BERITA. Strip pattern ini supaya model lebih general.
import re
data["text"] = data["text"].apply(
    lambda x: re.sub(r"^.*?\(.*?\)\s*[-–]?\s*", "", x)
)

# ── 4. Fix: shuffle data sebelum split ───────────────────────────
# concat tanpa shuffle → semua fake di atas, semua real di bawah
# model bisa bias urutan
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data["text"]
y = data["label"]

# ── 5. Fix: TF-IDF dengan parameter lebih robust ─────────────────
# max_features=5000 terlalu sempit → naikkan
# ngram_range=(1,2) → model tangkap frasa, bukan cuma kata tunggal
# min_df=2 → buang kata yang cuma muncul 1x (noise)
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y       # Fix: pastikan ratio fake/real seimbang di train & test
)

# ── 6. Fix: tambah class_weight balanced ─────────────────────────
# Kalau jumlah fake vs real tidak seimbang di dataset,
# model akan bias ke kelas mayoritas
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    C=1.0
)

model.fit(X_train, y_train)

# ── 7. Evaluasi lebih detail ──────────────────────────────────────
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("=" * 50)
print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print("=" * 50)
print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=["FAKE NEWS", "REAL NEWS"]))

# ── 8. Simpan model ───────────────────────────────────────────────
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\n✅ Model dan vectorizer berhasil disimpan!")
print("   → model/fake_news_model.pkl")
print("   → model/tfidf_vectorizer.pkl")