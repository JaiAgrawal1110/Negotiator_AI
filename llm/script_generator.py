"""
llm/script_generator.py
Week 7-8 — LLM Script Generation
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from groq import Groq

from llm.action_templates import get_strategy
from llm.market_retriever import MarketRateRetriever

GROQ_MODEL = "llama-3.3-70b-versatile"

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
    text = _CONTRACTION_GLUE.sub(r"\1 \2", text)
    text = _WORD_GLUE.sub(r"\1 \2", text)
    return text


@dataclass
class DealContext:
    freelancer_floor: float
    freelancer_target: float
    current_offer: float
    turn_number: int
    turns_remaining: int
    archetype: str
    client_last_message: str
    detected_sentiment: Optional[str] = None
    project_description: str = "a freelance project"
    currency: str = "$"
    extra_notes: str = field(default_factory=str)
    client_history: Optional[dict] = None


class ScriptGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GROQ_MODEL,
        use_market_context: bool = True,
    ):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model = model

        self.retriever = None
        if use_market_context:
            try:
                self.retriever = MarketRateRetriever()
            except Exception as e:
                print(f"⚠️  Market retriever unavailable ({e}). "
                      f"Continuing without market-rate grounding.")

    def _suggest_anchor_number(self, action_id: int, ctx: DealContext) -> Optional[float]:
        if action_id == 1:
            return round(ctx.freelancer_target * 0.96 / 25) * 25

        if action_id == 2:
            anchor = ctx.freelancer_target - 0.4 * (ctx.freelancer_target - ctx.current_offer)
            return round(anchor / 25) * 25

        return None

    def _get_market_context(self, ctx: DealContext) -> Optional[dict]:
        if self.retriever is None:
            return None
        try:
            return self.retriever.get_market_context(ctx.project_description, k=5)
        except Exception as e:
            print(f"⚠️  Market retrieval failed for this turn ({e}). Continuing without it.")
            return None

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
            "- CURRENCY: every number you state MUST use the exact currency "
            "symbol already present in freelancer_floor, freelancer_target, "
            "client_current_offer, or suggested_number as given to you (e.g. "
            "if you're given '₹99999', you say ₹99999 -- never $, never £, "
            "never any symbol other than the one already in the value). Do "
            "not convert to a different currency, do not substitute a symbol "
            "you find more familiar, and do not add or drop a symbol. This "
            "applies even if retrieved_market_data or your own phrasing "
            "would naturally read better in a different currency -- it "
            "would not, the client is quoting in the currency they were "
            "given and a switched symbol reads as a translation error.\n"
            "- If deal_context includes a 'suggested_number' field, you MUST use "
            "that exact number (rounding to the nearest 25 is fine) as the price "
            "you state. Do not substitute a different number, do not split the "
            "difference yourself, and do not invent your own anchor — the number "
            "has already been decided for you.\n"
            "- If there is no 'suggested_number' field, do not invent a new price "
            "at all — reference client_current_offer only if needed.\n"
            "- freelancer_floor and freelancer_target are given to you for your OWN "
            "awareness of the negotiation's boundaries — they are NEVER numbers to "
            "state directly as 'my price' or 'my rate'. The only number you may "
            "ever say out loud is suggested_number (when present) or "
            "client_current_offer (when referencing their offer back to them). "
            "Stating freelancer_target verbatim as your ask is a violation even if "
            "it happens to equal a number that sounds reasonable in context.\n"
            "- If deal_context includes 'retrieved_market_data', you may reference "
            "it in your justification (e.g. 'that's in line with market rate for "
            "similar backend API work') to make your reasoning more credible. "
            "This is supporting evidence only — it does NOT change what number you "
            "state. The number you state is still governed entirely by the "
            "'suggested_number'/'client_current_offer' rules above, even if "
            "retrieved_market_data's range differs from that number.\n"
            "- If deal_context includes 'client_history', you're negotiating with a "
            "REPEAT client -- you may let that inform your tone/confidence (e.g. "
            "slightly warmer if they've closed deals fairly before, more firm "
            "upfront if they've previously failed to close or lowballed "
            "repeatedly), but do not fabricate specific past quotes, dates, or "
            "conversations that aren't in the given history. If client_history "
            "is absent, treat this as a new client -- don't imply any prior "
            "relationship.\n"
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

        market_context = self._get_market_context(ctx)
        if market_context is not None:
            deal_context["retrieved_market_data"] = {
                "note": "Real freelancer rate data for similar work, for "
                        "justification/credibility only -- NOT the number "
                        "to state (use suggested_number or "
                        "client_current_offer for that, per the rules above).",
                "typical_rate_range": (
                    f"{ctx.currency}{market_context['suggested_range'][0]:g}"
                    f"-{ctx.currency}{market_context['suggested_range'][1]:g}/hr"
                ),
                "closest_comparable_project": market_context["closest_match"],
                "category": market_context["closest_category"],
            }

        if ctx.client_history is not None:
            deal_context["client_history"] = ctx.client_history

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
    # Numeric safety guard
    # ------------------------------------------------------------------
    def _allowed_numbers(self, action_id: int, ctx: DealContext) -> set:
        """The only numeric values the model is allowed to state, given
        this turn's action. Mirrors the system prompt's rules in code --
        prompt instructions alone weren't reliable enough (see: a real
        ₹99,999 hallucinated on an add_value turn with target ₹20,000).
        Floor/target are deliberately excluded: those are the freelancer's
        own boundaries, never numbers to state directly."""
        allowed = {round(ctx.current_offer, 2)}
        suggested = self._suggest_anchor_number(action_id, ctx)
        if suggested is not None:
            allowed.add(round(suggested, 2))
        return allowed

    def _find_violating_numbers(self, text: str, allowed: set, currency: str, tolerance: float = 1.0):
        """Extracts every currency-prefixed number in the generated text
        and returns any that don't match an allowed value within a small
        rounding tolerance. A non-empty result means the model stated a
        price nobody gave it."""
        pattern = re.escape(currency) + r"([\d,]+(?:\.\d+)?)"
        found = []
        for match in re.finditer(pattern, text):
            raw = match.group(1).replace(",", "")
            try:
                value = float(raw)
            except ValueError:
                continue
            if not any(abs(value - a) <= tolerance for a in allowed):
                found.append(value)
        return found

    def generate(self, action_id: int, ctx: DealContext, n_variants: int = 2) -> list[str]:
        """Returns a list of `n_variants` ready-to-send reply strings for the
        given action_id + deal context. Validates every returned script
        against the numbers it was actually allowed to state, retrying on
        a hallucinated price rather than silently handing a freelancer a
        number nobody computed."""
        allowed = self._allowed_numbers(action_id, ctx)
        max_attempts = 3
        last_violations = {}

        for attempt in range(1, max_attempts + 1):
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

            last_violations = {}
            for s in scripts:
                violations = self._find_violating_numbers(s, allowed, ctx.currency)
                if violations:
                    last_violations[s] = violations

            if not last_violations:
                return scripts

            print(f"⚠️  Script generator attempt {attempt}/{max_attempts} stated "
                  f"unauthorized number(s) {last_violations} (allowed: {allowed}). "
                  f"{'Retrying...' if attempt < max_attempts else 'Out of retries.'}")

        raise RuntimeError(
            f"Script generator repeatedly stated numbers it wasn't given "
            f"({last_violations}) after {max_attempts} attempts. Allowed "
            f"values were {allowed}. Refusing to return an unverified script "
            f"rather than risk sending a freelancer a hallucinated price."
        )

    def generate_single(self, action_id: int, ctx: DealContext) -> str:
        return self.generate(action_id, ctx, n_variants=1)[0]