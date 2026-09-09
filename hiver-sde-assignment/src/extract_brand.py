import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "twcs.csv"
OUTPUT_PATH = BASE_DIR / "data" / "amazon_help.csv"


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # Select AmazonHelp conversations
    amazon_df = df[df["author_id"] == "AmazonHelp"].copy()

    print(f"AmazonHelp messages: {len(amazon_df)}")

    # Save the extracted data
    amazon_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()