"""
scripts/inspect_market_data.py
Week 9-10 — Step 1: Inspect the raw dataset before cleaning it.

Kaggle's dataset pages are bot-walled, so the exact column names in
"Freelancer Earnings & Job Trends" (or whichever dataset you download)
couldn't be confirmed ahead of time. Rather than guess and hand you a
cleaning script that breaks on the first run, run this first — it just
prints what you actually have, so clean_market_data.py's COLUMN_MAP can
be filled in with real column names instead of assumptions.

Usage:
    1. Download the CSV from Kaggle, drop it in data/raw_market_data.csv
    2. python -m scripts.inspect_market_data
    3. Use the printed output to fill in COLUMN_MAP at the top of
       clean_market_data.py
"""

import pandas as pd

RAW_PATH = "data/raw_market_data.csv"


def inspect():
    df = pd.read_csv(RAW_PATH)

    print(f"Loaded {len(df)} rows from {RAW_PATH}\n")

    print("--- Columns + dtypes ---")
    print(df.dtypes)

    print("\n--- First 5 rows ---")
    print(df.head().to_string())

    print("\n--- Null counts per column ---")
    print(df.isnull().sum())

    # Try to flag likely rate/price columns and likely category/skill
    # columns by name, as a hint (not a guarantee) for filling COLUMN_MAP.
    rate_like = [c for c in df.columns if any(
        k in c.lower() for k in ["rate", "price", "earn", "salary", "pay", "hourly", "budget"]
    )]
    category_like = [c for c in df.columns if any(
        k in c.lower() for k in ["category", "skill", "job", "title", "type", "field", "domain"]
    )]

    print(f"\n--- Columns that might be RATE data: {rate_like or 'none detected'} ---")
    print(f"--- Columns that might be CATEGORY/SKILL data: {category_like or 'none detected'} ---")

    if rate_like:
        print("\n--- Summary stats on likely rate column(s) ---")
        for col in rate_like:
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"\n{col}:")
                print(df[col].describe())


if __name__ == "__main__":
    inspect()
