"""
scripts/generate_market_chunks.py
Week 9-10 — Step 3: Turn structured rate data into embeddable RAG chunks.

ONE-TIME BATCH JOB, same pattern as nlp/generate_data.py from Week 5-6.
This calls Groq once per (category, experience_level) combo to generate
varied, realistic project-description phrasings tied to that combo's real
rate range from market_rates_summary.csv. The output gets embedded in a
later step (Step 4) and stored in a vector store — after that, nothing in
the live negotiation pipeline calls Groq for market data again. Retrieval
at negotiation time is a local vector similarity search only.

Each chunk pairs a natural-language project description (the thing that
gets embedded and semantically matched against a real negotiation's
project description) with the real numeric rate data behind it (the thing
that actually gets used once retrieval finds a match).

Usage:
    python -m scripts.generate_market_chunks

Output:
    data/market_rag_chunks.csv
"""

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
SUMMARY_PATH = "data/market_rates_summary.csv"
OUTPUT_PATH = "data/market_rag_chunks.csv"

VARIANTS_PER_COMBO = 10  # 24 combos x 10 = ~240 chunks, comparable in
                          # scale to the Week 5-6 sentiment data generation run

SYSTEM_PROMPT = (
    "You generate realistic freelance project descriptions for a dataset "
    "used in a negotiation coaching tool. Given a job category and "
    "experience level, write short, varied, realistic descriptions of "
    "specific projects a freelancer at that level might take on in that "
    "category — the kind of description a client would actually give when "
    "posting a job (skills required, deliverable type, scale), not a "
    "generic label.\n\n"
    "Rules:\n"
    "- Each description should be 1-2 sentences, specific enough to be "
    "useful for matching against a real client's project description.\n"
    "- Vary the sub-type within the category across the set (e.g. within "
    "'Web Development' vary between backend APIs, e-commerce sites, "
    "admin dashboards, etc. — don't write 10 near-identical descriptions).\n"
    "- Do not mention rates, prices, or dollar amounts in the description "
    "itself — that data is attached separately.\n"
    "- Output strict JSON only, no markdown fences, no commentary."
)


def _build_user_prompt(category: str, level: str, n: int) -> str:
    return json.dumps({
        "instruction": f"Generate {n} distinct project descriptions.",
        "job_category": category,
        "experience_level": level,
        "output_format": {"descriptions": ["string, one project description"]},
    })


def generate_chunks_for_combo(client: Groq, category: str, level: str, n: int) -> list[str]:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(category, level, n)},
        ],
        temperature=0.9,  # higher than script_generator.py -- we want
                           # variety across descriptions here, not precision
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)
    return parsed["descriptions"]


def generate():
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    summary = pd.read_csv(SUMMARY_PATH)

    all_chunks = []
    chunk_id = 0

    for i, row in summary.iterrows():
        category = row["Job_Category"]
        level = row["Experience_Level"]
        print(f"[{i + 1}/{len(summary)}] Generating {VARIANTS_PER_COMBO} chunks for "
              f"{category} / {level}...")

        try:
            descriptions = generate_chunks_for_combo(client, category, level, VARIANTS_PER_COMBO)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ⚠️  Failed on {category}/{level}: {e}. Skipping this combo.")
            continue

        for desc in descriptions:
            all_chunks.append({
                "chunk_id": chunk_id,
                "job_category": category,
                "experience_level": level,
                "project_description": desc,
                "rate_median": row["median"],
                "rate_mean": row["mean"],
                "rate_min": row["min"],
                "rate_max": row["max"],
                "sample_count": row["count"],
            })
            chunk_id += 1

        # Save incrementally after every combo, not just at the end --
        # if this crashes partway through (rate limit, network blip), you
        # keep everything generated so far instead of losing the whole run.
        pd.DataFrame(all_chunks).to_csv(OUTPUT_PATH, index=False)

        time.sleep(0.5)  # light pacing, avoid hammering the rate limit

    print(f"\nDone. Saved {len(all_chunks)} chunks to {OUTPUT_PATH}")
    print(f"({summary.shape[0]} combos x ~{VARIANTS_PER_COMBO} variants each)")


if __name__ == "__main__":
    generate()
