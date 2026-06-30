"""
Groq-powered client simulator.

This is the LLM "mouth" for the client side: given an archetype, deal
context, and conversation history, it generates the next human-sounding
client message in character.

Used for:
  - Demos / qualitative testing of archetype behavior
  - Generating the actual training transcripts shown to users
  - Few-shot grounding examples for the archetype classifier (Week 5-6)

NOT used inside the hot RL training loop — see env/negotiation_env.py
for why (speed/cost). The rule-based env handles price/state transitions
at training time; this module produces the matching dialogue when needed.

Usage:
    from llm.simulator import ClientSimulator

    sim = ClientSimulator()
    reply = sim.get_client_reply(
        archetype="lowballer",
        deal_context={...},  # see archetypes.get_archetype_prompt() kwargs
        history=[{"role": "freelancer", "content": "..."}],
    )
"""

import os
from groq import Groq
from dotenv import load_dotenv

from env.archetypes import get_archetype_prompt, ARCHETYPES

load_dotenv()

DEFAULT_MODEL = "llama-3.1-70b-versatile"
BACKUP_MODEL = "mixtral-8x7b-32768"


class ClientSimulator:
    def __init__(self, api_key: str = None, model: str = DEFAULT_MODEL):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key or key == "your_key_here":
            raise ValueError(
                "GROQ_API_KEY not set. Get a free key at console.groq.com "
                "and put it in your .env file."
            )
        self.client = Groq(api_key=key)
        self.model = model

    def get_client_reply(
        self,
        archetype: str,
        deal_context: dict,
        history: list[dict],
        model: str = None,
    ) -> str:
        """
        archetype: one of env.archetypes.ARCHETYPES keys
        deal_context: kwargs for get_archetype_prompt (project_type, skill,
            quoted_rate, market_rate_low, market_rate_high, budget_ceiling,
            deadline, turn_number)
        history: list of {"role": "freelancer"|"client", "content": str},
            in chronological order. Last entry should be the freelancer's
            most recent message.
        """
        if archetype not in ARCHETYPES:
            raise ValueError(f"Unknown archetype '{archetype}'. Choose from: {list(ARCHETYPES.keys())}")

        system_prompt = get_archetype_prompt(archetype, **deal_context)

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            # Client's own past messages map to "assistant", freelancer's to "user"
            role = "assistant" if turn["role"] == "client" else "user"
            messages.append({"role": role, "content": turn["content"]})

        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=0.8,
                max_tokens=200,
            )
        except Exception:
            # fall back to backup model if primary is unavailable/rate-limited
            response = self.client.chat.completions.create(
                model=BACKUP_MODEL,
                messages=messages,
                temperature=0.8,
                max_tokens=200,
            )

        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Manual smoke test — requires a real GROQ_API_KEY in .env
    sim = ClientSimulator()
    context = dict(
        project_type="internal admin dashboard",
        skill="full-stack development",
        quoted_rate="$5,000",
        market_rate_low="$4,500",
        market_rate_high="$6,500",
        budget_ceiling="$5,800",
        deadline="4 weeks",
        turn_number=1,
    )
    history = [
        {"role": "freelancer", "content": "Hi! Following up on the dashboard project — my quote is $5,000 for the full build, 4 week turnaround."}
    ]
    reply = sim.get_client_reply("lowballer", context, history)
    print("CLIENT (lowballer):", reply)
