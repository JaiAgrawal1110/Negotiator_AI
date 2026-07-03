"""
api/main.py
Week 11-12 — FastAPI backend

Wraps the existing pipeline (RL policy -> sentiment -> script generator ->
market RAG -> memory) behind HTTP endpoints for the React frontend.

IMPORTANT DIFFERENCE FROM scripts/demo_negotiation.py:
demo_negotiation.py runs against NegotiationEnv, which simulates the CLIENT
side with rule-based logic (see env/negotiation_env.py's own docstring --
that's a training/demo convenience, not a live negotiation partner). Here,
the client is a REAL person typing real messages, so this module does NOT
import or step NegotiationEnv at all. It reimplements just the pieces of
env state that a live session actually needs (price_position, turn
tracking, relationship_score bookkeeping) and builds the same 13-float
observation vector shape the trained PPO policy expects, so the same
policy file works unmodified in production.

Fields the original env samples randomly during training (leverage_score,
deadline_urgency) have no equivalent live signal yet -- there's no sensor
for "how much leverage does this freelancer have." For now these are
supplied by the freelancer at session start (defaulting to a neutral 0.5
if omitted) rather than invented by the backend. Revisit if a real signal
for these becomes available.

Archetype: there's no live archetype classifier wired into the project
yet (the transcript mentions a "few-shot archetype classifier" as existing
somewhere, but that file wasn't available to build against). Until it
exists, the freelancer selects/confirms the archetype at session start,
same as passing `archetype=` into env.reset() during testing. Swapping in
a real classifier later only touches the /negotiation/start handler --
everything downstream (obs vector, policy, script generation) is
unaffected.

Run:
    uvicorn api.main:app --reload
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env.negotiation_env import ACTIONS, ARCHETYPE_LIST
from memory.negotiation_memory import NegotiationMemory
from llm.script_generator import ScriptGenerator, DealContext

MODEL_PATH = "agent/policy/ppo_negotiation.zip"

ARCHETYPE_DISPLAY_NAMES = {
    "lowballer": "Lowballer",
    "ghoster": "Ghoster",
    "friendly_crusher": "Friendly Crusher",
    "deadline_rusher": "Deadline Rusher",
    "scope_creeper": "Scope Creeper",
}

# ---------------------------------------------------------------------- #
# Singletons — loaded once at process start, not per-request
# ---------------------------------------------------------------------- #

app = FastAPI(title="Negotiator API")

app.add_middleware(
    CORSMiddleware,
    # Vite picks the next free port if its default is taken (5173, 5174,
    # 5175...), so pin this to a regex covering any localhost/127.0.0.1
    # port rather than an exact list -- avoids CORS breaking every time
    # a stray dev server is already holding 5173. Scoped to localhost
    # only; tighten this to a real origin before deploying anywhere else.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory = NegotiationMemory()
generator = ScriptGenerator()

_policy = None  # lazy-loaded, see get_policy()


def get_policy():
    """Loads the trained PPO policy once and caches it. Lazy rather than
    at import time so the API can still start (e.g. for local frontend
    dev against stubbed responses) if the model file isn't present yet --
    it'll just fail on first /step call with a clear error instead of
    refusing to boot at all."""
    global _policy
    if _policy is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"No trained policy found at {MODEL_PATH}. "
                       f"Run `python -m agent.train` first.",
            )
        from stable_baselines3 import PPO
        _policy = PPO.load(MODEL_PATH)
    return _policy


_sentiment_classifier = None
_sentiment_unavailable = False  # sticky flag so we don't retry-load every request


def get_sentiment_classifier():
    """Same graceful-degradation pattern already used elsewhere in this
    project (market retriever, script generator): sentiment is a nice-to-
    have signal, not a hard dependency. If the model hasn't been trained
    yet (or the seed dataset makes it unreliable, per nlp/sentiment.py's
    own honest note), fall back to a neutral 0.5 rather than 500ing every
    negotiation step over it."""
    global _sentiment_classifier, _sentiment_unavailable
    if _sentiment_unavailable:
        return None
    if _sentiment_classifier is None:
        try:
            from nlp.sentiment import ToneClassifier
            _sentiment_classifier = ToneClassifier()
        except Exception as e:
            print(f"⚠️  Sentiment classifier unavailable ({e}). "
                  f"Falling back to neutral sentiment for this process.")
            _sentiment_unavailable = True
            return None
    return _sentiment_classifier


# ---------------------------------------------------------------------- #
# Session state (in-memory for now -- swap for Redis/DB before multi-
# worker deployment; a single uvicorn worker is fine for the frontend
# milestone but this dict won't survive a restart or scale past one
# process)
# ---------------------------------------------------------------------- #

@dataclass
class SessionState:
    session_id: str
    client_id: str
    archetype: str  # snake_case, one of ARCHETYPE_LIST
    floor: float
    target: float
    current_offer: float
    project_description: str
    project_category: Optional[str]
    max_turns: int
    currency: str = "$"
    turn: int = 0
    relationship_score: float = 0.7
    leverage_score: float = 0.5
    deadline_urgency: float = 0.5
    last_offer_delta: float = 0.0
    sentiment_score: float = 0.5
    prev_offer: float = field(default=0.0)
    done: bool = False


SESSIONS: dict[str, SessionState] = {}


def _get_session(session_id: str) -> SessionState:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'.")
    if session.done:
        raise HTTPException(
            status_code=409,
            detail=f"Session '{session_id}' already ended. Start a new one.",
        )
    return session


def _build_obs(session: SessionState) -> np.ndarray:
    """Same 13-float layout as NegotiationEnv._get_obs() -- see that
    module's docstring for the field-by-field spec. Kept in sync manually
    since this backend doesn't import/step the env; if that layout ever
    changes, this function needs the matching update."""
    price_position = float(np.clip(
        (session.current_offer - session.floor)
        / max(session.target - session.floor, 1e-6),
        0.0,
        1.0,
    ))
    archetype_onehot = [1.0 if session.archetype == a else 0.0 for a in ARCHETYPE_LIST]
    return np.array(
        [
            price_position,
            session.turn / session.max_turns,
            (session.max_turns - session.turn) / session.max_turns,
            *archetype_onehot,
            session.leverage_score,
            session.relationship_score,
            session.deadline_urgency,
            session.last_offer_delta,
            session.sentiment_score,
        ],
        dtype=np.float32,
    )


def _sentiment_label(score: float) -> str:
    if score < 0.35:
        return "negative/resistant"
    if score > 0.65:
        return "positive/collaborative"
    return "neutral"


def _client_history_dict(client_id: str) -> Optional[dict]:
    summary = memory.get_client_summary(client_id)
    if summary is None:
        return None
    return {
        "past_negotiation_count": summary.past_negotiation_count,
        "deals_closed": summary.deals_closed,
        "deals_lost": summary.deals_lost,
        "avg_pct_of_target_on_closed_deals": summary.avg_pct_of_target,
        "last_outcome": summary.last_outcome,
    }


# ---------------------------------------------------------------------- #
# Request / response schemas
# ---------------------------------------------------------------------- #

class StartRequest(BaseModel):
    client_id: str
    archetype: str = Field(..., description="One of: " + ", ".join(ARCHETYPE_LIST))
    floor: float
    target: float
    current_offer: float = Field(..., description="Client's opening offer.")
    project_description: str
    project_category: Optional[str] = None
    max_turns: int = 8
    leverage_score: float = 0.5
    deadline_urgency: float = 0.5
    currency: str = Field(
        "$", description="Currency symbol for amounts in scripts and the UI, e.g. '$' or '₹'."
    )


class StartResponse(BaseModel):
    session_id: str
    client_history: Optional[dict]
    state: dict


class StepRequest(BaseModel):
    client_message: str = Field(..., description="Real client's latest message, verbatim.")
    client_offer: Optional[float] = Field(
        None, description="Client's numeric offer this turn, if it changed. "
                           "Omit if the client's message didn't restate a number."
    )
    n_variants: int = Field(1, ge=1, le=4)


class StepResponse(BaseModel):
    turn: int
    action_id: int
    action_name: str
    scripts: list[str]
    detected_sentiment: str
    state: dict


class EndRequest(BaseModel):
    final_deal: Optional[float] = Field(
        None, description="Final agreed price, or None if no deal was reached."
    )


class EndResponse(BaseModel):
    saved: bool
    client_id: str
    final_deal: Optional[float]
    deal_closed: bool


def _state_dict(session: SessionState) -> dict:
    return {
        "turn": session.turn,
        "max_turns": session.max_turns,
        "current_offer": round(session.current_offer, 2),
        "floor": session.floor,
        "target": session.target,
        "relationship_score": round(session.relationship_score, 2),
        "archetype": ARCHETYPE_DISPLAY_NAMES.get(session.archetype, session.archetype),
        "currency": session.currency,
        "done": session.done,
    }


# ---------------------------------------------------------------------- #
# Endpoints
# ---------------------------------------------------------------------- #

@app.post("/negotiation/start", response_model=StartResponse)
def start_negotiation(req: StartRequest):
    if req.archetype not in ARCHETYPE_LIST:
        raise HTTPException(
            status_code=422,
            detail=f"archetype must be one of {ARCHETYPE_LIST}, got '{req.archetype}'.",
        )

    session_id = str(uuid.uuid4())
    session = SessionState(
        session_id=session_id,
        client_id=req.client_id,
        archetype=req.archetype,
        floor=req.floor,
        target=req.target,
        current_offer=req.current_offer,
        project_description=req.project_description,
        project_category=req.project_category,
        max_turns=req.max_turns,
        leverage_score=req.leverage_score,
        deadline_urgency=req.deadline_urgency,
        currency=req.currency,
        prev_offer=req.current_offer,
    )
    SESSIONS[session_id] = session

    return StartResponse(
        session_id=session_id,
        client_history=_client_history_dict(req.client_id),
        state=_state_dict(session),
    )


@app.post("/negotiation/{session_id}/step", response_model=StepResponse)
def step_negotiation(session_id: str, req: StepRequest):
    session = _get_session(session_id)
    policy = get_policy()

    # 1. Update numeric state with the client's latest offer, if given.
    if req.client_offer is not None:
        session.last_offer_delta = float(np.clip(
            (req.client_offer - session.current_offer)
            / max(session.target - session.floor, 1e-6),
            -1.0,
            1.0,
        ))
        session.prev_offer = session.current_offer
        session.current_offer = req.client_offer
    else:
        session.last_offer_delta = 0.0

    # 2. Score sentiment on the real message text, falling back to neutral.
    classifier = get_sentiment_classifier()
    if classifier is not None:
        try:
            session.sentiment_score = classifier.sentiment_score_for_state_vector(
                req.client_message
            )
        except Exception as e:
            print(f"⚠️  Sentiment scoring failed this turn ({e}). Using previous value.")
    detected_sentiment = _sentiment_label(session.sentiment_score)

    session.turn += 1

    # 3. Policy picks the action for this turn.
    obs = _build_obs(session)
    action_id, _ = policy.predict(obs, deterministic=True)
    action_id = int(action_id)

    # 4. Turn that into ready-to-send script(s).
    ctx = DealContext(
        freelancer_floor=session.floor,
        freelancer_target=session.target,
        current_offer=session.current_offer,
        turn_number=session.turn,
        turns_remaining=session.max_turns - session.turn,
        archetype=ARCHETYPE_DISPLAY_NAMES.get(session.archetype, session.archetype),
        client_last_message=req.client_message,
        detected_sentiment=detected_sentiment,
        project_description=session.project_description,
        currency=session.currency,
        extra_notes=f"Turn {session.turn} of {session.max_turns}. "
                    f"Relationship score: {round(session.relationship_score, 2)}.",
        client_history=_client_history_dict(session.client_id),
    )
    scripts = generator.generate(action_id, ctx, n_variants=req.n_variants)

    if session.turn >= session.max_turns:
        session.done = True

    return StepResponse(
        turn=session.turn,
        action_id=action_id,
        action_name=ACTIONS[action_id],
        scripts=scripts,
        detected_sentiment=detected_sentiment,
        state=_state_dict(session),
    )


@app.post("/negotiation/{session_id}/end", response_model=EndResponse)
def end_negotiation(session_id: str, req: EndRequest):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'.")

    memory.save_episode(
        client_id=session.client_id,
        archetype=session.archetype,
        floor=session.floor,
        target=session.target,
        final_deal=req.final_deal,
        turns_taken=session.turn,
        project_category=session.project_category,
        reward=None,  # reward is a training-time concept; no env/reward fn live here
    )
    session.done = True

    return EndResponse(
        saved=True,
        client_id=session.client_id,
        final_deal=req.final_deal,
        deal_closed=req.final_deal is not None,
    )


@app.get("/negotiation/{session_id}/state", response_model=dict)
def get_state(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'.")
    return _state_dict(session)


@app.get("/clients/{client_id}/history", response_model=dict)
def get_client_history(client_id: str, limit: int = 10):
    summary = memory.get_client_summary(client_id)
    raw = memory.get_raw_history(client_id, limit=limit)
    return {
        "summary": _client_history_dict(client_id) if summary else None,
        "recent_negotiations": raw,
    }


@app.get("/health")
def health():
    return {"status": "ok"}