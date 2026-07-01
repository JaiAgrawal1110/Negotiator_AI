"""
llm/script_generator.py
Week 7-8 — LLM Script Generation

This is the layer that makes the agent "visibly useful": it takes the
abstract action the RL policy picked (0-5) plus the current deal state,
client archetype, and detected sentiment/tone, and turns it into an
actual message text a freelancer could copy-paste and send.

Pipeline position:
    RL policy (agent/) --> action_id --------\
    NLP sentiment (nlp/) --> tone signal ------> ScriptGenerator --> reply text(s)
    Archetype classifier (nlp/) --> archetype -/
    NegotiationEnv state --> price/turn context /

Uses the same Groq client pattern as llm/simulator.py and nlp/classifier.py
(model swapped to llama-3.3-70b-versatile after the 3.1 decommission).
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from groq import Groq

from llm.action_templates import get_strategy

GROQ_MODEL = "llama-3.3-70b-versatile"

# Common glued-word patterns seen from Groq/Llama output where a space gets
# dropped at a word boundary (e.g. "Let'sexplore" -> "Let's explore",
# "withthe" -> "with the", "valueit" -> "value it"). Two shapes handled:
#   1) a contraction glued directly to the next word ("it'sbest")
#   2) a content word glued directly to a short function word ("valueit")
# Conservative on purpose — the function-word list is short, so it won't
# split legitimate longer words that happen to end in one of these letters.
_CONTRACTION_GLUE = re.compile(
    r"\b(Let's|let's|It's|it's|That's|that's|I'm|I've|I'll|You're|you're)([a-z])"
)

_GLUE_PREFIXES = ["with", "value", "willing", "going", "trying", "happy",
                   "able", "ready", "looking", "hoping"]
_GLUE_SUFFIXES = ["the", "it", "to", "is", "are", "of", "on", "at"]
_WORD_GLUE = re.compile(
    r"\b(" + "|".join(_GLUE_PREFIXES) + r")(" + "|".join(_GLUE_SUFFIXES) + r")\b"
)


def _fix_glued_words(text: str) -> str:
    """Best-effort cleanup for missing-space artifacts. Not a substitute for
    the prompt instruction — a second layer of defense, since the model
    doesn't reliably self-catch this in generation. The prefix/suffix lists
    are small on purpose; extend them if new glue patterns show up in
    testing rather than trying to catch every case up front."""
    text = _CONTRACTION_GLUE.sub(r"\1 \2", text)
    text = _WORD_GLUE.sub(r"\1 \2", text)
    return text


@dataclass
class DealContext:
    """Snapshot of the negotiation at the current turn. Mirrors the fields
    already tracked in env/negotiation_env.py's state vector — pull these
    straight from env.state / env.info when wiring this into a live episode."""

    freelancer_floor: float
    freelancer_target: float
    current_offer: float          # client's latest number on the table
    turn_number: int
    turns_remaining: int
    archetype: str                # one of: Lowballer, Ghoster, Friendly Crusher,
                                   # Deadline Rusher, Scope Creeper
    client_last_message: str      # what the client just said (for grounding tone)
    detected_sentiment: Optional[str] = None   # from nlp/sentiment.py, e.g. "aggressive"
    project_description: str = "a freelance project"
    currency: str = "$"
    extra_notes: str = field(default_factory=str)  # e.g. "3rd counter-offer this thread"


class ScriptGenerator:
    """Wraps Groq to turn (action_id, DealContext) into freelancer reply scripts."""

    def __init__(self, api_key: Optional[str] = None, model: str = GROQ_MODEL):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model = model

    def _suggest_anchor_number(self, action_id: int, ctx: DealContext) -> Optional[float]:
        """Computes the actual number the script should state, where the
        strategy calls for one. Numbers are decided deterministically here —
        the LLM's job is to phrase them, not invent them. This is the same
        split used elsewhere in the project: rule-based logic drives numbers,
        the LLM drives text/tone only.

        Returns None when the strategy shouldn't assert a new number at all
        (e.g. Hold and Reframe usually just references the existing offer).
        """
        if action_id == 1:  # Re-anchor Higher — push toward the real target,
            # not a midpoint. Leave a small amount of room to negotiate
            # further rather than opening at the exact ceiling.
            return round(ctx.freelancer_target * 0.96 / 25) * 25

        if action_id == 2:  # Concede Partially — move from target toward
            # the client's offer, but only partway (60% of the way from
            # target to their offer, biased to still favor the freelancer).
            anchor = ctx.freelancer_target - 0.4 * (ctx.freelancer_target - ctx.current_offer)
            return round(anchor / 25) * 25

        # Actions 0 (Hold), 3 (Add Value), 4 (Boundary), 5 (Walk) reference
        # the existing current_offer rather than asserting a new number —
        # already working correctly in testing, no override needed.
        return None

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        return (
            "You are a negotiation coach writing reply scripts on behalf of a "
            "freelancer in an active client negotiation. You are NOT the client "
            "and NOT the freelancer's assistant persona — you write in the "
            "freelancer's voice, first person, ready to send as-is.\n\n"
            "Rules:\n"
            "- Write like a real person texting/emailing a client, not a lawyer "
            "or a customer service bot. No corporate filler, no 'I hope this "
            "email finds you well'.\n"
            "- Keep it short. 2-5 sentences unless the strategy explicitly needs more.\n"
            "- Never reveal you are following a 'strategy' or mention negotiation "
            "tactics by name.\n"
            "- If deal_context includes a 'suggested_number' field, you MUST use "
            "that exact number (rounding to the nearest 25 is fine) as the price "
            "you state. Do not substitute a different number, do not split the "
            "difference yourself, and do not invent your own anchor — the number "
            "has already been decided for you.\n"
            "- If there is no 'suggested_number' field, do not invent a new price "
            "at all — reference client_current_offer only if needed.\n"
            "- Never invent deadlines, urgency, or framing (e.g. 'high-priority') "
            "that isn't present in deal_context.\n"
            "- Match the tone guidance exactly — it is calibrated per strategy.\n"
            "- Proofread before returning: every word must be separated by a single "
            "space. A bad example to avoid: 'the valueit will bring' (missing space "
            "between 'value' and 'it'). Re-read your output character by character "
            "for this specific error before returning.\n"
            "- Output strict JSON only, no markdown fences, no commentary."
        )

    def _build_user_prompt(self, action_id: int, ctx: DealContext, n_variants: int) -> str:
        strategy = get_strategy(action_id)
        suggested_number = self._suggest_anchor_number(action_id, ctx)

        deal_context = {
            "project_description": ctx.project_description,
            "freelancer_floor": f"{ctx.currency}{ctx.freelancer_floor}",
            "freelancer_target": f"{ctx.currency}{ctx.freelancer_target}",
            "client_current_offer": f"{ctx.currency}{ctx.current_offer}",
            "turn_number": ctx.turn_number,
            "turns_remaining": ctx.turns_remaining,
            "client_archetype": ctx.archetype,
            "detected_client_sentiment": ctx.detected_sentiment or "unknown",
            "client_last_message": ctx.client_last_message,
            "extra_notes": ctx.extra_notes,
        }
        if suggested_number is not None:
            deal_context["suggested_number"] = f"{ctx.currency}{suggested_number:g}"

        return json.dumps({
            "instruction": (
                f"Write {n_variants} distinct reply script variant(s) the freelancer "
                f"can send right now, executing the strategy below."
            ),
            "strategy": {
                "name": strategy["name"],
                "goal": strategy["goal"],
                "tone_guidance": strategy["tone_guidance"],
                "avoid": strategy["avoid"],
            },
            "deal_context": deal_context,
            "output_format": {
                "scripts": [
                    {"variant": "string, the message text, ready to send"}
                ]
            },
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, action_id: int, ctx: DealContext, n_variants: int = 2) -> list[str]:
        """Returns a list of `n_variants` ready-to-send reply strings for the
        given action_id + deal context."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(action_id, ctx, n_variants)},
            ],
            temperature=0.8,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        try:
            parsed = json.loads(raw)
            scripts = [_fix_glued_words(s["variant"]) for s in parsed["scripts"]]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise RuntimeError(
                f"Script generator got malformed JSON from Groq: {raw!r}"
            ) from e

        if not scripts:
            raise RuntimeError("Script generator returned zero scripts.")

        return scripts

    def generate_single(self, action_id: int, ctx: DealContext) -> str:
        """Convenience wrapper for when you only need one script (e.g. wired
        into an automated pipeline rather than shown to the user as choices)."""
        return self.generate(action_id, ctx, n_variants=1)[0]