"""
memory/negotiation_memory.py
Week 9-10 — Memory (second half of the phase, after Market RAG).

Stores negotiation outcomes per CLIENT (not per archetype). Archetype is
a behavioral category ("Lowballer") shared by many different clients --
remembering "Lowballer clients tend to..." isn't the same as remembering
"this specific client, last time, opened low then came up fast." The
latter is what real repeat-client memory should mean, so this is keyed
by client_id.

Design note: nothing in the project currently has a real client identity
system (that arrives with Week 11-12's API/frontend, where actual client
accounts will exist). Until then, client_id is just a string the caller
supplies -- scripts/demo_negotiation.py passes one in for testing/demo
purposes. When the real API exists, it'll pass real client IDs through
the same interface with no changes needed here.

Uses SQLite (Python stdlib, no new dependency) rather than the vector
store from Market RAG -- this is structured, exact-match lookup by
client_id, not semantic search, so a vector store would be the wrong
tool here.
"""

import sqlite3
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

DB_PATH = "memory/negotiation_memory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS negotiation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    archetype TEXT NOT NULL,
    project_category TEXT,
    floor REAL NOT NULL,
    target REAL NOT NULL,
    final_deal REAL,             -- NULL if no deal reached
    deal_closed INTEGER NOT NULL,  -- 0/1
    turns_taken INTEGER NOT NULL,
    reward REAL,
    timestamp TEXT NOT NULL
);
"""


@dataclass
class ClientHistorySummary:
    client_id: str
    past_negotiation_count: int
    deals_closed: int
    deals_lost: int
    avg_pct_of_target: Optional[float]  # avg final_deal/target across closed deals, as a %
    most_common_archetype: str
    last_outcome: str  # human-readable, e.g. "closed at 94% of target 3 negotiations ago"


class NegotiationMemory:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------
    def save_episode(
        self,
        client_id: str,
        archetype: str,
        floor: float,
        target: float,
        final_deal: Optional[float],
        turns_taken: int,
        project_category: Optional[str] = None,
        reward: Optional[float] = None,
    ) -> None:
        """Call this once, right after an episode ends (terminated or
        truncated), with the same values already being printed in
        demo_negotiation.py's end-of-episode summary."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO negotiation_history
                   (client_id, archetype, project_category, floor, target,
                    final_deal, deal_closed, turns_taken, reward, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    client_id, archetype, project_category, floor, target,
                    final_deal, int(final_deal is not None), turns_taken,
                    reward, datetime.now(timezone.utc).isoformat(),
                ),
            )

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------
    def get_raw_history(self, client_id: str, limit: int = 10) -> list[dict]:
        """Most recent negotiations for this client, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM negotiation_history
                   WHERE client_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (client_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_client_summary(self, client_id: str) -> Optional[ClientHistorySummary]:
        """Rolls up this client's history into something ready to drop
        into DealContext.client_history. Returns None for a client with
        no prior history (the common case -- most negotiations are with
        a new client), so callers can distinguish "no data" from "bad data"
        rather than getting an empty-but-present summary either way."""
        history = self.get_raw_history(client_id, limit=50)
        if not history:
            return None

        deals_closed = sum(1 for h in history if h["deal_closed"])
        deals_lost = len(history) - deals_closed

        pct_of_targets = [
            (h["final_deal"] / h["target"]) * 100
            for h in history if h["deal_closed"] and h["target"]
        ]
        avg_pct = round(statistics.mean(pct_of_targets), 1) if pct_of_targets else None

        archetypes = [h["archetype"] for h in history]
        most_common = max(set(archetypes), key=archetypes.count)

        most_recent = history[0]
        if most_recent["deal_closed"]:
            pct = round((most_recent["final_deal"] / most_recent["target"]) * 100)
            last_outcome = f"closed at {pct}% of target"
        else:
            last_outcome = "no deal reached"

        return ClientHistorySummary(
            client_id=client_id,
            past_negotiation_count=len(history),
            deals_closed=deals_closed,
            deals_lost=deals_lost,
            avg_pct_of_target=avg_pct,
            most_common_archetype=most_common,
            last_outcome=last_outcome,
        )
