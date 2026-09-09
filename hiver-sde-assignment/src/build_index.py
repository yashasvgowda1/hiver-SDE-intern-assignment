import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

INPUT_FILE = "data/amazon_support.csv"
MODEL_DIR = "models"

print("Loading customer messages...")

df = pd.read_csv(INPUT_FILE)

df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)

print("Messages loaded:", len(df))

print("Building TF-IDF index...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["text"])

print("Vector shape:", X.shape)

model = NearestNeighbors(
    n_neighbors=5,
    metric="cosine"
)

model.fit(X)

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
joblib.dump(model, os.path.join(MODEL_DIR, "search_model.pkl"))

df[["tweet_id", "author_id", "text"]].to_pickle(
    os.path.join(MODEL_DIR, "messages.pkl")
)

print()
print("Index built successfully!")
print("Saved:")
print("models/vectorizer.pkl")
print("models/search_model.pkl")
print("models/messages.pkl")