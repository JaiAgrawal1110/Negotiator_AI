"""
Groq-powered synthetic training data generator.

Generates 300 varied examples per tone class (2,400 total) using
Llama 3.1 70B, anchored to the real examples already in seed_training_data.py
so synthetic examples match the same register and context.

Saves directly into nlp/generated_training_data.py — then sentiment.py
automatically picks up both seed + generated data for fine-tuning.

Run: python -m nlp.generate_data
     python -m nlp.generate_data --per-class 500  (for more data)

After this finishes, re-run: python -m nlp.sentiment
"""

import os
import json
import time
import argparse
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

LABELS = [
    "hesitant",
    "urgent",
    "aggressive",
    "bluffing",
    "collaborative",
    "dismissive",
    "enthusiastic",
    "noncommittal",
]

# Real anchor examples per class — shown to the LLM so it generates
# in the same register (freelance/client negotiation, not generic chat)
REAL_ANCHORS = {
    "hesitant": [
        "This seems extremely high. Let me get back to you on Monday.",
        "I'm a little unsure about committing to that number right now.",
        "That's a bit more than I was expecting, I don't know.",
    ],
    "urgent": [
        "I need you to respond to my questions. Asap. And be communicating.",
        "We're launching Monday, can you confirm today if you can take this on?",
        "This is time-sensitive, I need an answer ASAP.",
    ],
    "aggressive": [
        "I need productivity throughout your whole shift. I'll be cutting pay rate if standards are not met.",
        "From now on three strike rule going into effect.",
        "why is that so high? it's just a headshot? it's not even real art because it's digital lol no offense",
    ],
    "bluffing": [
        "How about 80 that's as high as I can go.",
        "Our most senior Data Analyst in Delhi earns this amount working full time with 20 years of experience.",
        "I've got three other freelancers ready to do this for half the price.",
    ],
    "collaborative": [
        "We shall do 3 months at INR 25k and then switch to INR 30k after 3 months. Please let me know if this works for you.",
        "We want you to feel comfortable with the cost. Come to us with a proposal.",
        "Let's figure out something that works well for both of us.",
    ],
    "dismissive": [
        "I don't understand. You want us to pay your taxes too?",
        "Normally when we agree on an amount the taxes is on the person receiving it. This is how payments work.",
        "I'm paying you 4.25 rn.",
    ],
    "enthusiastic": [
        "This is exactly what we've been looking for, love it!",
        "Your portfolio is amazing, I'm really excited to work with you on this.",
        "Yes! Let's do this, I'm genuinely thrilled about the direction.",
    ],
    "noncommittal": [
        "Maybe think of an average monthly salary for the next 3 months. Then we review and revise in 3 months?",
        "You suggest something and let me know.",
        "We'll see, I'll get back to you at some point.",
    ],
}

TONE_DESCRIPTIONS = {
    "hesitant": "uncertain, cautious, needs more time, not ready to commit, slightly nervous about the price or scope",
    "urgent": "pressuring for speed, tight deadline, needs answer now, time-sensitive, creating time pressure",
    "aggressive": "hostile, threatening, belittling the freelancer's work or rate, demanding, confrontational",
    "bluffing": "using fake leverage, name-dropping competitors, claiming budget limits that aren't real, comparing to irrelevant benchmarks",
    "collaborative": "warm, solution-oriented, wants to find middle ground, genuinely trying to make the deal work for both sides",
    "dismissive": "brushing off concerns, doesn't value the freelancer's perspective, condescending, treating the issue as trivial",
    "enthusiastic": "excited, complimentary, keen to move forward, positive about the freelancer's work",
    "noncommittal": "vague, deferring, won't give a clear answer, buying time without saying so directly",
}

BATCH_SIZE = 20  # examples per API call
SLEEP_BETWEEN_CALLS = 1.2  # seconds, to stay under Groq rate limits


def generate_batch(client: Groq, tone: str, n: int, batch_num: int) -> list[str]:
    anchors = "\n".join(f'- "{ex}"' for ex in REAL_ANCHORS[tone])
    description = TONE_DESCRIPTIONS[tone]

    prompt = f"""You are generating training data for a negotiation tone classifier.

Generate exactly {n} short client messages (1-3 sentences each) that clearly show the tone: {tone.upper()}

Tone definition: {description}

These messages are from a CLIENT negotiating with a FREELANCER over project rate, scope, or payment.
Context: freelance work (software dev, design, content, general freelancing).

Real examples of this tone to anchor your style:
{anchors}

RULES:
- Each message must be distinct — vary the wording, situation, and phrasing significantly
- Keep it realistic — like a real WhatsApp or email message from a client
- Do NOT label or number the messages
- Do NOT include any preamble or explanation
- Output ONLY a JSON array of strings, nothing else
- Example format: ["message one", "message two", "message three"]

Generate {n} {tone} messages now:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85,
        max_tokens=1500,
    )

    raw = response.choices[0].message.content.strip()

    # strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    messages = json.loads(raw)
    return [m.strip() for m in messages if m.strip()]


def generate_all(per_class: int = 300, output_path: str = None):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError("GROQ_API_KEY not set in .env")

    client = Groq(api_key=api_key)

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "generated_training_data.py")

    all_data = []

    for tone in LABELS:
        print(f"\nGenerating {per_class} examples for: {tone.upper()}")
        tone_examples = []
        remaining = per_class
        batch_num = 0

        while remaining > 0:
            batch_n = min(BATCH_SIZE, remaining)
            try:
                batch = generate_batch(client, tone, batch_n, batch_num)
                tone_examples.extend(batch)
                remaining -= len(batch)
                batch_num += 1
                print(f"  batch {batch_num}: got {len(batch)} examples "
                      f"({per_class - remaining}/{per_class} done)")
                time.sleep(SLEEP_BETWEEN_CALLS)
            except Exception as e:
                print(f"  ⚠️  batch {batch_num} failed: {e} — retrying in 3s")
                time.sleep(3)
                continue

        all_data.extend([(msg, tone) for msg in tone_examples[:per_class]])
        print(f"  ✅ {tone}: {min(len(tone_examples), per_class)} examples collected")

    # write output file
    lines = [
        '"""',
        f'Auto-generated training data — {len(all_data)} examples across {len(LABELS)} tone classes.',
        f'Generated via Groq/Llama 3.1 70B, anchored to real freelance negotiation examples.',
        f'DO NOT hand-edit — re-run nlp/generate_data.py to regenerate.',
        '"""',
        '',
        'GENERATED_DATA = [',
    ]
    for msg, tone in all_data:
        escaped = msg.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'    ("{escaped}", "{tone}"),')
    lines.append(']')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n✅ Done. {len(all_data)} examples saved to {output_path}")
    print("Now run: python -m nlp.sentiment")
    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=300,
                        help="Examples to generate per tone class (default 300, total = 300x8 = 2400)")
    args = parser.parse_args()
    generate_all(per_class=args.per_class)
