"""
Client archetype classifier — few-shot LLM classification via Groq.

After 2-3 messages in a negotiation, classifies the client into one of
the 5 archetypes (env/archetypes.py) so the right RL strategy/policy
context can be applied. Few-shot prompting (not a fine-tuned model) is
used here per the blueprint, since archetype signals are more about
behavioral pattern recognition across a short conversation than single-
message tone classification (that's what nlp/sentiment.py handles).

Output includes a confidence score so the calling code (Week 11-12 API)
can decide whether to commit to a strategy or ask for one more turn of
signal before classifying.

Run a manual smoke test: python -m nlp.classifier
(requires a real GROQ_API_KEY in .env)
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

from env.archetypes import ARCHETYPES

load_dotenv()

ARCHETYPE_LIST = list(ARCHETYPES.keys())

FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Client messages so far:
1. "We love your work but honestly Rs.65,000 feels really high for us, could you do something closer to Rs.40,000?"
Classification: {"archetype": "lowballer", "confidence": 0.82, "reasoning": "Opens with a steep discount ask (~40% below quote) immediately, framed as budget constraint, while praising the work -- classic anchor-low-and-test pattern."}

EXAMPLE 2:
Client messages so far:
1. "This all sounds great, let's move forward!"
2. (after freelancer sends contract/invoice) [3 days pass with no reply]
3. "Hey sorry, been super busy, let me look at this properly soon."
Classification: {"archetype": "ghoster", "confidence": 0.78, "reasoning": "Enthusiastic agreement followed by unexplained silence after a commitment point, then a vague low-effort excuse -- avoidance pattern typical of stalling for leverage."}

EXAMPLE 3:
Client messages so far:
1. "Your portfolio is incredible, we'd love to work with you!"
2. "This is amazing, just one small thing -- could we trim the price just a touch to make it work on our end?"
Classification: {"archetype": "friendly_crusher", "confidence": 0.75, "reasoning": "Heavy compliments paired with a soft, 'small' price-down ask -- using warmth to make a discount feel reasonable rather than negotiating directly."}

EXAMPLE 4:
Client messages so far:
1. "We need this live by end of week, can you confirm today if you can take this on?"
Classification: {"archetype": "deadline_rusher", "confidence": 0.7, "reasoning": "Leads immediately with a tight deadline and demands a same-day confirmation -- using urgency to compress the freelancer's decision time."}

EXAMPLE 5:
Client messages so far:
1. "Sounds good, let's go with the agreed scope and price."
2. (after kickoff) "Oh, and could you also just quickly add a login page too? Should be simple."
Classification: {"archetype": "scope_creeper", "confidence": 0.73, "reasoning": "Agreed to original scope/price readily, then introduces a new deliverable framed as minor/obvious immediately after kickoff -- classic incremental scope expansion."}
"""

SYSTEM_PROMPT = f"""You are a negotiation pattern classifier. Given a short
sequence of a CLIENT's messages in a freelance project negotiation, classify
which of these 5 client archetypes best matches the behavior so far:

{', '.join(ARCHETYPE_LIST)}

Definitions:
- lowballer: anchors well below market rate, testing if freelancer caves
- ghoster: goes silent after receiving a counter, uses absence as pressure
- friendly_crusher: warm and complimentary, but chips price down via small asks
- deadline_rusher: creates urgency to rush the freelancer's decision
- scope_creeper: agrees to terms then slowly adds requirements

Here are labeled examples of the classification format:
{FEW_SHOT_EXAMPLES}

Respond ONLY with a JSON object: {{"archetype": "...", "confidence": 0.0-1.0, "reasoning": "..."}}
No markdown, no preamble, JSON only. If signal is too thin/ambiguous after
only 1 message, you may still give your best guess but lower the confidence
accordingly (e.g. below 0.5).
"""


class ArchetypeClassifier:
    def __init__(self, api_key: str = None, model: str = "llama-3.1-70b-versatile"):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key or key == "your_key_here":
            raise ValueError(
                "GROQ_API_KEY not set. Get a free key at console.groq.com "
                "and put it in your .env file."
            )
        self.client = Groq(api_key=key)
        self.model = model

    def classify(self, client_messages: list[str]) -> dict:
        """
        client_messages: chronological list of the client's messages so far
        (freelancer messages excluded -- we're classifying CLIENT behavior).

        Returns: {"archetype": str, "confidence": float, "reasoning": str}
        """
        numbered = "\n".join(f"{i+1}. \"{m}\"" for i, m in enumerate(client_messages))
        user_prompt = f"Client messages so far:\n{numbered}\n\nClassification:"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,  # low temp -- this is classification, not creative generation
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # defensive fallback if the model wraps output in markdown fences anyway
            cleaned = raw.strip("`").replace("json\n", "").strip()
            result = json.loads(cleaned)

        if result.get("archetype") not in ARCHETYPE_LIST:
            raise ValueError(f"Model returned unknown archetype: {result.get('archetype')}")

        return result


if __name__ == "__main__":
    clf = ArchetypeClassifier()
    messages = [
        "We love your work but Rs.65,000 is above our budget. Can you do Rs.42,000?",
    ]
    result = clf.classify(messages)
    print(json.dumps(result, indent=2))
