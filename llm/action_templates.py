"""
llm/action_templates.py
Week 7-8 — LLM Script Generation

Maps the RL agent's 6 discrete action IDs (defined in env/negotiation_env.py)
to the strategic intent + tone guidance the LLM needs to turn an abstract
action into an actual message a freelancer would send.

The RL agent decides WHAT to do (the action ID). This file tells the LLM
HOW that action should sound in a real negotiation message.
"""

# Must match the action space in env/negotiation_env.py exactly.
ACTION_STRATEGIES = {
    0: {
        "name": "Hold and Reframe",
        "goal": (
            "Do not move on price. Reframe the conversation around value, "
            "scope, or outcomes instead of haggling over the number directly."
        ),
        "tone_guidance": (
            "Calm, confident, non-defensive. Acknowledge what the client said, "
            "then pivot to value/outcomes. No apologizing for the price."
        ),
        "avoid": "Don't concede anything. Don't repeat the same justification twice.",
    },
    1: {
        "name": "Re-anchor Higher",
        "goal": (
            "Push the number back up, or reintroduce a higher anchor point than "
            "the client's current offer, backed by a concrete reason (market rate, "
            "scope, comparable rate, complexity)."
        ),
        "tone_guidance": (
            "Assertive but collaborative — never aggressive. State the number "
            "plainly, back it with one reason, and stop talking."
        ),
        "avoid": "Don't over-explain or sound like you're negotiating against yourself.",
    },
    2: {
        "name": "Concede Partially",
        "goal": (
            "Move toward the client's number, but only partway, and always attach "
            "the concession to something in return (a condition, a trade, a smaller "
            "scope, a shorter timeline, or a trial structure)."
        ),
        "tone_guidance": (
            "Warm and solution-oriented, but not passive. Frame the concession as a "
            "deliberate offer, not a retreat."
        ),
        "avoid": "Never concede without attaching a condition. Never apologize for the ask.",
    },
    3: {
        "name": "Add Value (Non-Price)",
        "goal": (
            "Keep price where it is, but sweeten the deal with something that costs "
            "little relative to its value to the client — faster turnaround, an extra "
            "revision round, a bonus deliverable, priority support, etc."
        ),
        "tone_guidance": "Generous but strategic — this is a trade, not a giveaway.",
        "avoid": "Don't imply the original price was too high; this is an add-on, not a discount.",
    },
    4: {
        "name": "Set Boundary / Firm No",
        "goal": (
            "Clearly decline the client's ask (price, scope, deadline, or terms) "
            "without leaving room for further pressure, while keeping the door open "
            "for them to come back with something workable."
        ),
        "tone_guidance": (
            "Direct, respectful, unemotional. Short sentences. No hedging language "
            "like 'maybe' or 'I guess'."
        ),
        "avoid": "Don't soften it into a maybe. Don't over-justify — one reason is enough.",
    },
    5: {
        "name": "Walk Away / End Negotiation",
        "goal": (
            "End the negotiation professionally, either because terms can't be met "
            "or because the client's behavior (ghosting, disrespect, lowballing past "
            "the floor) makes the deal not worth pursuing."
        ),
        "tone_guidance": (
            "Professional, unemotional, brief. No burning bridges, no guilt-tripping, "
            "no last-minute discount to save the deal."
        ),
        "avoid": "Don't leave the offer open. Don't soften it into a negotiation move.",
    },
}


def get_strategy(action_id: int) -> dict:
    """Fetch the strategy block for a given RL action ID, with a safety check."""
    if action_id not in ACTION_STRATEGIES:
        raise ValueError(
            f"Unknown action_id={action_id}. Must be 0-5, matching "
            f"env/negotiation_env.py's action space."
        )
    return ACTION_STRATEGIES[action_id]
