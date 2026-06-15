import os
import gdown

# Mapping file → Google Drive ID
models = {
    "model/word2vec.pkl":           "16uSilSWohzvzrfZTsztsgGLEnokuFk6p",
    "model/tfidf_vectorizer.pkl":   "1-dsHMx6YYctQH0EzKLG9-EnWFLr7IzJs",
    "model/nlm_classifier.pkl":     "1QNgRYlv9ImdE7TGFaS2nRB04-_yPgikG",
    "model/fake_news_model.pkl":    "1Yq4-W35zijS2af9PTYsRYkGU5YRkb310",
}

os.makedirs("model", exist_ok=True)

for filename, file_id in models.items():
    if not os.path.exists(filename):
        print(f"⬇️  Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
        print(f"✅ {filename} berhasil didownload!")
    else:
        print(f"✅ {filename} sudah ada, skip.")

print("\n🎉 Semua model siap!")