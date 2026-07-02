"""
scripts/build_vector_store.py
Week 9-10 — Step 4a: Embed the RAG chunks and build the vector store.

ONE-TIME BATCH JOB (like Step 3) — no API calls here at all, since
sentence-transformers runs locally. Run this once after
generate_market_chunks.py produces data/market_rag_chunks.csv, and again
any time that file changes (new chunks added, data refreshed).

Nothing in the live negotiation pipeline calls this script directly —
market_retriever.py (Step 4b) reads the persisted store this script builds.

Usage:
    python -m scripts.build_vector_store
"""

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/market_rag_chunks.csv"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "market_rates"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build():
    df = pd.read_csv(CHUNKS_PATH)
    print(f"Loaded {len(df)} chunks from {CHUNKS_PATH}")

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads "
          f"~80MB, cached locally after that)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Embedding {len(df)} project descriptions...")
    embeddings = model.encode(
        df["project_description"].tolist(),
        show_progress_bar=True,
    )

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop and rebuild the collection each run, so re-running this script
    # after regenerating chunks doesn't leave stale/duplicate entries behind.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet, nothing to drop
    collection = client.create_collection(COLLECTION_NAME)

    # Chroma metadata values must be str/int/float/bool -- no NaNs, no
    # numpy types. Cast explicitly rather than relying on pandas dtypes.
    metadatas = [
        {
            "job_category": str(row["job_category"]),
            "experience_level": str(row["experience_level"]),
            "rate_median": float(row["rate_median"]),
            "rate_mean": float(row["rate_mean"]),
            "rate_min": float(row["rate_min"]),
            "rate_max": float(row["rate_max"]),
            "sample_count": int(row["sample_count"]),
        }
        for _, row in df.iterrows()
    ]

    collection.add(
        ids=[str(cid) for cid in df["chunk_id"]],
        embeddings=embeddings.tolist(),
        documents=df["project_description"].tolist(),
        metadatas=metadatas,
    )

    print(f"\nBuilt vector store at {CHROMA_PATH} with {collection.count()} entries "
          f"in collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build()
