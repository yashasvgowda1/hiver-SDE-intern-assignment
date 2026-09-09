import pandas as pd
import joblib

MODEL_DIR = "models"

# Load saved files
vectorizer = joblib.load(f"{MODEL_DIR}/vectorizer.pkl")
model = joblib.load(f"{MODEL_DIR}/search_model.pkl")
messages = pd.read_pickle(f"{MODEL_DIR}/messages.pkl")


def search_customer_messages(query, top_k=5):
    query_vector = vectorizer.transform([query])

    distances, indices = model.kneighbors(
        query_vector,
        n_neighbors=top_k
    )

    results = []

    for distance, index in zip(distances[0], indices[0]):
        row = messages.iloc[index]

        results.append({
            "tweet_id": row["tweet_id"],
            "author_id": row["author_id"],
            "text": row["text"],
            "similarity": round(1 - distance, 4)
        })

    return results


print("====================================")
print("   CUSTOMER SUPPORT SEARCH SYSTEM")
print("====================================")

while True:

    query = input("\nEnter customer complaint (or 'exit'): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    results = search_customer_messages(query)

    print("\nSimilar customer messages:\n")

    for i, result in enumerate(results, 1):

        print(f"{i}. Similarity: {result['similarity']}")
        print(f"   Tweet ID: {result['tweet_id']}")
        print(f"   Author: {result['author_id']}")
        print(f"   Message: {result['text']}")
        print("-" * 70)