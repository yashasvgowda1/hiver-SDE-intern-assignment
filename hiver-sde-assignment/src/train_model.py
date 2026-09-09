import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# =========================
# FILE PATHS
# =========================

DATA_FILE = "data/intent_clusters.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "intent_model.pkl")


# =========================
# LOAD DATA
# =========================

print("Loading intent dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# =========================
# CHECK COLUMNS
# =========================

required_columns = ["text", "cluster"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in CSV!"
        )


# =========================
# CLEAN DATA
# =========================

df = df.dropna(subset=["text", "cluster"])

df["text"] = df["text"].astype(str)
df["cluster"] = df["cluster"].astype(str)


print("Usable rows:", len(df))
print("Number of clusters:", df["cluster"].nunique())


# =========================
# DATA
# =========================

X = df["text"]
y = df["cluster"]


# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# =========================
# MACHINE LEARNING MODEL
# =========================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# =========================
# TRAIN
# =========================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# =========================
# EVALUATE
# =========================

print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL ACCURACY")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# =========================
# SAVE MODEL
# =========================

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, MODEL_FILE)

print("\n==============================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================")

print("Model:", MODEL_FILE)