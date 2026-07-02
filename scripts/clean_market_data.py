"""
scripts/clean_market_data.py
Week 9-10 — Step 2: Clean the raw Kaggle dataset into RAG-ready market data.

Built against the confirmed real schema of "Freelancer Earnings & Job
Trends" (1,950 rows, no nulls):

    Freelancer_ID, Job_Category, Platform, Experience_Level, Client_Region,
    Payment_Method, Job_Completed, Earnings_USD, Hourly_Rate,
    Job_Success_Rate, Client_Rating, Job_Duration_Days, Project_Type,
    Rehire_Rate, Marketing_Spend

We only need a subset of these for market-rate retrieval. Hourly_Rate is
the actual per-hour benchmark (5.02-99.83 range observed); Earnings_USD is
total earnings across all completed jobs, not a per-project rate, so it's
dropped rather than confused with a rate signal.

Produces two outputs:
  1. data/market_rates_cleaned.csv   - row-level, one listing per row,
     used as individual retrievable chunks for the RAG embedding step.
  2. data/market_rates_summary.csv   - aggregated by (Job_Category,
     Experience_Level), used for quick lookups and to spot which
     category/experience combos are thin on real data (< MIN_SAMPLES),
     which is exactly what the Groq synthetic expansion step (Step 3)
     needs to target.

Usage:
    python -m scripts.clean_market_data
"""

import pandas as pd

RAW_PATH = "data/raw_market_data.csv"
CLEANED_PATH = "data/market_rates_cleaned.csv"
SUMMARY_PATH = "data/market_rates_summary.csv"

# Below this many real samples for a (category, experience) combo, flag it
# as needing synthetic expansion in Step 3 rather than trusting the real
# data alone — a median from 3 rows isn't a reliable market rate.
MIN_SAMPLES = 15

KEEP_COLUMNS = [
    "Job_Category",
    "Platform",
    "Experience_Level",
    "Client_Region",
    "Hourly_Rate",
    "Job_Success_Rate",
    "Client_Rating",
    "Project_Type",
]


def clean():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} raw rows.")

    df = df[KEEP_COLUMNS].copy()

    # Drop rows with clearly broken rates (<=0 or absurdly high — the
    # observed real range tops out at ~$100/hr, so treat >$500/hr as a
    # data error rather than a legitimate outlier worth keeping).
    before = len(df)
    df = df[(df["Hourly_Rate"] > 0) & (df["Hourly_Rate"] <= 500)]
    print(f"Dropped {before - len(df)} rows with invalid Hourly_Rate.")

    # Normalize category/experience text (strip whitespace, consistent case)
    # so groupby doesn't silently split "Web Development " from
    # "Web Development" into two buckets.
    df["Job_Category"] = df["Job_Category"].str.strip()
    df["Experience_Level"] = df["Experience_Level"].str.strip()

    df.to_csv(CLEANED_PATH, index=False)
    print(f"Saved {len(df)} cleaned rows to {CLEANED_PATH}")

    # --- Aggregated summary per (category, experience) ---
    summary = (
        df.groupby(["Job_Category", "Experience_Level"])["Hourly_Rate"]
        .agg(count="count", median="median", mean="mean", min="min", max="max", std="std")
        .reset_index()
        .sort_values(["Job_Category", "Experience_Level"])
    )
    summary["needs_synthetic_expansion"] = summary["count"] < MIN_SAMPLES

    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Saved {len(summary)} category/experience combos to {SUMMARY_PATH}")

    thin = summary[summary["needs_synthetic_expansion"]]
    print(f"\n{len(thin)} of {len(summary)} combos have fewer than "
          f"{MIN_SAMPLES} real samples — these are Step 3's synthetic "
          f"expansion targets:")
    if not thin.empty:
        print(thin[["Job_Category", "Experience_Level", "count"]].to_string(index=False))

    print(f"\nUnique Job_Category values ({df['Job_Category'].nunique()}): "
          f"{sorted(df['Job_Category'].unique())}")
    print(f"Unique Experience_Level values ({df['Experience_Level'].nunique()}): "
          f"{sorted(df['Experience_Level'].unique())}")


if __name__ == "__main__":
    clean()
