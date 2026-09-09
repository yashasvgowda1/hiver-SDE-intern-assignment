import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "amazon_help.csv"


def main():
    print("Loading AmazonHelp data...")

    df = pd.read_csv(DATA_PATH)

    print("\nTotal AmazonHelp messages:", len(df))

    print("\nInbound values:")
    print(df["inbound"].value_counts(dropna=False))

    print("\nSample AmazonHelp messages:\n")

    for i, text in enumerate(df["text"].dropna().head(20), 1):
        print(f"{i}. {text}")


if __name__ == "__main__":
    main()