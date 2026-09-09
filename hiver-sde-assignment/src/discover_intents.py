import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/amazon_support.csv"
N_CLUSTERS = 10


# ============================================================
# LOAD AMAZON SUPPORT DATASET
# ============================================================

print("Loading Amazon support dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded!")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# CHECK TEXT COLUMN
# ============================================================

if "text" not in df.columns:
    raise ValueError(
        "Column 'text' was not found in amazon_support.csv"
    )

df = df.dropna(subset=["text"]).copy()

df["text"] = df["text"].astype(str)


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["clean_text"] = df["text"].apply(clean_text)

df = df[df["clean_text"].str.len() > 5].copy()

print("Usable customer messages:", len(df))


# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF representation...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5
)

X = vectorizer.fit_transform(df["clean_text"])


# ============================================================
# CLUSTER CUSTOMER MESSAGES
# ============================================================

print("Discovering customer-support intents...")

model = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

df["cluster"] = model.fit_predict(X)


# ============================================================
# SHOW IMPORTANT WORDS FOR EACH CLUSTER
# ============================================================

terms = vectorizer.get_feature_names_out()

print("\n")
print("=" * 70)
print("DISCOVERED CUSTOMER SUPPORT INTENTS")
print("=" * 70)

for cluster_number in range(N_CLUSTERS):

    cluster_messages = df[df["cluster"] == cluster_number]

    center = model.cluster_centers_[cluster_number]

    top_indices = center.argsort()[-10:][::-1]

    keywords = [
        terms[index]
        for index in top_indices
    ]

    print(f"\nINTENT {cluster_number + 1}")
    print("-" * 50)

    print("Keywords:")
    print(", ".join(keywords))

    print("Number of messages:", len(cluster_messages))

    print("\nExample customer messages:")

    examples = cluster_messages["text"].head(3)

    for example in examples:
        print("-", example[:250])


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = "data/intent_clusters.csv"

df[
    ["tweet_id", "author_id", "text", "cluster"]
].to_csv(
    output_file,
    index=False
)

print("\n")
print("=" * 70)
print("DONE!")
print("=" * 70)
print("Saved:", output_file)
print("Total messages:", len(df))
print("Number of discovered intents:", N_CLUSTERS)