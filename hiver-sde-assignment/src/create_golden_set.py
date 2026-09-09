import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/amazon_support.csv")
OUTPUT_FILE = Path("evaluation/golden_set.csv")

N_EXAMPLES = 200

print("Loading Amazon support dataset...")

df = pd.read_csv(INPUT_FILE)

# Keep real customer messages with usable text
df = df.dropna(subset=["text"]).copy()
df["text"] = df["text"].astype(str)

df = df[df["text"].str.len() >= 20]

# Reproducible sample
golden = df.sample(
    n=N_EXAMPLES,
    random_state=42
).copy()

# These MUST be manually labelled.
golden["intent"] = ""
golden["expected_action"] = ""
golden["label_reason"] = ""

# Save
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

golden[
    [
        "tweet_id",
        "text",
        "intent",
        "expected_action",
        "label_reason"
    ]
].to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nGolden set created successfully!")
print("Number of examples:", len(golden))
print("Saved to:", OUTPUT_FILE)

print("\nIMPORTANT:")
print("The intent, expected_action and label_reason columns")
print("must be completed by hand.")