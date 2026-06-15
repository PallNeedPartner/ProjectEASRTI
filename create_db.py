import sqlite3
import os

os.makedirs("database", exist_ok=True)

conn   = sqlite3.connect("database/news.db")
cursor = conn.cursor()

# Buat tabel dengan kolom model_used (tracking NLP vs NLM)
cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    news_text  TEXT,
    prediction TEXT,
    confidence REAL,
    model_used TEXT DEFAULT 'NLP (TF-IDF)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Kalau tabel sudah ada tapi belum punya kolom model_used
# (migrasi dari versi lama), tambahkan kolomnya
try:
    cursor.execute("ALTER TABLE history ADD COLUMN model_used TEXT DEFAULT 'NLP (TF-IDF)'")
    print("✅ Kolom model_used ditambahkan ke tabel existing.")
except sqlite3.OperationalError:
    pass  # Kolom sudah ada, skip

conn.commit()
conn.close()

print("✅ Database siap!")