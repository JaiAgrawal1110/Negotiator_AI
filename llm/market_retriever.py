"""
llm/market_retriever.py
Week 9-10 — Step 4b: Runtime market-rate retrieval.

This is the piece that actually runs during a live negotiation. It does
NOT call Groq or any API — it's a local embedding + vector similarity
lookup against the store build_vector_store.py already built. Fast,
free, and offline once the store exists on disk.

Usage:
    retriever = MarketRateRetriever()
    result = retriever.get_market_context("build a REST API for a booking app")
    # result["suggested_median"], result["suggested_range"], result["matches"]
"""

import statistics
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "market_rates"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class MarketMatch:
    project_description: str
    job_category: str
    experience_level: str
    rate_median: float
    rate_min: float
    rate_max: float
    sample_count: int
    distance: float  # lower = more semantically similar


class MarketRateRetriever:
    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.model = SentenceTransformer(model_name)
        client = chromadb.PersistentClient(path=chroma_path)
        try:
            self.collection = client.get_collection(collection_name)
        except Exception as e:
            raise RuntimeError(
                f"No vector store found at {chroma_path} (collection "
                f"'{collection_name}'). Run `python -m scripts.build_vector_store` "
                f"first."
            ) from e

    def retrieve(
        self,
        project_description: str,
        k: int = 5,
        experience_level: Optional[str] = None,
    ) -> list[MarketMatch]:
        """Returns the k most semantically similar market-rate chunks.

        experience_level: optionally filter to a specific level (e.g.
        matching the freelancer's own level) rather than mixing Beginner
        and Expert rates for the same type of work.
        """
        query_embedding = self.model.encode([project_description])[0].tolist()

        where = {"experience_level": experience_level} if experience_level else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
        )

        if not results["ids"][0]:
            return []

        matches = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            matches.append(MarketMatch(
                project_description=results["documents"][0][i],
                job_category=meta["job_category"],
                experience_level=meta["experience_level"],
                rate_median=meta["rate_median"],
                rate_min=meta["rate_min"],
                rate_max=meta["rate_max"],
                sample_count=meta["sample_count"],
                distance=results["distances"][0][i],
            ))
        return matches

    def get_market_context(
        self,
        project_description: str,
        k: int = 5,
        experience_level: Optional[str] = None,
    ) -> Optional[dict]:
        """Convenience wrapper: retrieves matches and rolls them up into a
        single summary ready to drop into DealContext.extra_notes or a new
        dedicated field. Returns None if the store has no matches at all
        (e.g. empty store, or an overly narrow experience_level filter).

        suggested_range is built from the spread of the retrieved MEDIANS
        (a tight, defensible band), not the min/max of the underlying raw
        data each match came from — the latter balloons to nearly the
        dataset's full span once you combine several matches (each
        category/level combo's own min-max range is already wide, e.g.
        $7-$100 for one single combo), which produced a useless
        "market rate is $6-$100" result in testing. Median-of-medians +
        interquartile spread gives a much more usable number.
        """
        matches = self.retrieve(project_description, k=k, experience_level=experience_level)
        if not matches:
            return None

        medians = sorted(m.rate_median for m in matches)
        center = statistics.median(medians)

        if len(medians) >= 4:
            q1, q3 = statistics.quantiles(medians, n=4)[0], statistics.quantiles(medians, n=4)[2]
        else:
            # too few matches for a meaningful quartile split -- fall back
            # to the observed min/max of the medians themselves (still far
            # tighter than the old min/max-of-raw-ranges behavior).
            q1, q3 = medians[0], medians[-1]

        return {
            "suggested_median": round(center, 2),
            "suggested_range": (round(q1, 2), round(q3, 2)),
            "closest_match": matches[0].project_description,
            "closest_category": matches[0].job_category,
            "matches": matches,
        }