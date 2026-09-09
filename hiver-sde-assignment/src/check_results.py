import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_FILE = BASE_DIR / "evaluation" / "golden_set.csv"

print("Loading golden set...")

if not GOLDEN_FILE.exists():
    print("ERROR: golden_set.csv not found!")
    print(f"Expected location: {GOLDEN_FILE}")
    exit()

df = pd.read_csv(GOLDEN_FILE)

print("\nGolden set loaded successfully!")
print("Number of examples:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nSample examples:")
print(df.head(10).to_string(index=False))

print("\nDone!")