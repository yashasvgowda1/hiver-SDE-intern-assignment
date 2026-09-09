import pandas as pd
import os

# Paths
input_path = "data/twcs.csv"
output_path = "data/amazon_customer_messages.csv"

print("Loading dataset...")

df = pd.read_csv(input_path)

print("Dataset loaded!")
print("Total rows:", len(df))
print("Columns:", list(df.columns))

# Convert inbound column to proper boolean
df["inbound"] = df["inbound"].astype(str).str.lower().map({
    "true": True,
    "false": False
})

# Customer messages are inbound messages
customer_df = df[df["inbound"] == True].copy()

# Keep only useful columns
customer_df = customer_df[["tweet_id", "text"]]

# Remove empty messages
customer_df = customer_df.dropna(subset=["text"])
customer_df = customer_df[customer_df["text"].str.strip() != ""]

print("Customer messages found:", len(customer_df))

# Save
customer_df.to_csv(output_path, index=False)

print("Saved successfully!")
print("Output:", output_path)

print("\nSample customer messages:")
print(customer_df.head(10))