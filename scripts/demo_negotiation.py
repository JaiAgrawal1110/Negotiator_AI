"""
scripts/demo_negotiation.py
Week 7-8 — LLM Script Generation demo/integration harness

Two modes:

1. STANDALONE (default, no dependencies on your trained PPO model) — lets you
   test script_generator.py in isolation by manually stepping through actions.
   Good for sanity-checking prompt quality before wiring anything else up.

2. FULL PIPELINE (set USE_TRAINED_AGENT = True) — loads your trained PPO
   policy from Week 3-4 and lets it pick the action_id at each turn, then
   calls the script generator to turn that into real text. This is the
   "agent actually plays the negotiation and writes the message" milestone.

Run:
    python -m scripts.demo_negotiation
"""

import os
from dotenv import load_dotenv
load_dotenv()
from llm.script_generator import ScriptGenerator, DealContext

USE_TRAINED_AGENT = False  # flip to True once you've trained a policy via agent/train.py
MODEL_PATH = "agent/policy/ppo_negotiation.zip"  # matches train.py's POLICY_PATH exactly

# env/negotiation_env.py uses lowercase snake_case archetype keys internally
# (ARCHETYPE_LIST); this maps them to the display names action_templates.py
# and the earlier standalone tests were written against.
ARCHETYPE_DISPLAY_NAMES = {
    "lowballer": "Lowballer",
    "ghoster": "Ghoster",
    "friendly_crusher": "Friendly Crusher",
    "deadline_rusher": "Deadline Rusher",
    "scope_creeper": "Scope Creeper",
}


MARKET_CHUNKS_PATH = "data/market_rag_chunks.csv"
_market_chunks_cache = None  # loaded once, lazily, not per-episode


def _sample_project_description(category: str) -> str:
    """Pulls a real project description for the given category from the
    240 chunks generated in Week 9-10 Step 3 (generate_market_chunks.py).
    Falls back to a generic label if that file doesn't exist yet, rather
    than crashing the whole demo over a missing optional file."""
    global _market_chunks_cache
    if _market_chunks_cache is None:
        try:
            import pandas as pd
            _market_chunks_cache = pd.read_csv(MARKET_CHUNKS_PATH)
        except FileNotFoundError:
            _market_chunks_cache = False  # sentinel: tried and failed, don't retry

    if _market_chunks_cache is False:
        return f"a {category.lower()} project"

    matches = _market_chunks_cache[_market_chunks_cache["job_category"] == category]
    if matches.empty:
        return f"a {category.lower()} project"

    return matches.sample(1).iloc[0]["project_description"]


def _sentiment_label(sentiment_score: float) -> str:
    """env.sentiment_score is a raw 0-1 float (real NLP wiring lands in
    Week 5-6's nlp/sentiment.py output, not produced by the env itself).
    Bucket it into a coarse label so the script generator has something
    readable rather than a bare number."""
    if sentiment_score < 0.35:
        return "negative/resistant"
    if sentiment_score > 0.65:
        return "positive/collaborative"
    return "neutral"


def _describe_client_offer(current_offer: float, previous_offer: float) -> str:
    """NegotiationEnv is deliberately numbers-only during training (see its
    own docstring — real dialogue text is generated separately by
    llm/simulator.py, not inside the hot training loop). So there's no real
    client message to hand the script generator here. This synthesizes a
    plain, honest description of the numeric move instead of inventing fake
    dialogue — swap this out for llm/simulator.py's actual output once
    that's wired in, for a real quote instead of a description of one."""
    if previous_offer is None or current_offer == previous_offer:
        return f"Client's current offer on the table is ${current_offer:g}."
    direction = "raised their offer to" if current_offer > previous_offer else "lowered their offer to"
    return f"Client {direction} ${current_offer:g} (was ${previous_offer:g})."


def run_full_pipeline_demo(archetype: str = None, max_turns: int = 8):
    """Loads the trained PPO agent, lets it pick action_id at each turn of
    a real episode, and calls the script generator on each one — the actual
    "agent plays a negotiation and writes its own messages" milestone.

    archetype: one of ARCHETYPE_DISPLAY_NAMES' keys (e.g. "lowballer"), or
               None to let the env sample one at random like it does during
               training.
    """
    from stable_baselines3 import PPO
    from env.negotiation_env import NegotiationEnv, ACTIONS

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run `python -m agent.train` "
            f"first, or set USE_TRAINED_AGENT = False to use the standalone demo."
        )

    model = PPO.load(MODEL_PATH)
    generator = ScriptGenerator()

    env = NegotiationEnv(max_turns=max_turns)
    reset_options = {"archetype": archetype} if archetype else {}
    obs, info = env.reset(options=reset_options)

    # Sample ONE real project description matching this episode's actual
    # category (from the 240 chunks generated in Week 9-10 Step 3), reused
    # for the whole episode -- the project doesn't change mid-negotiation,
    # so re-sampling every turn would be wrong. This replaces the old
    # generic "a freelance project" placeholder, which gave the market
    # retriever nothing real to match against.
    project_description = _sample_project_description(info["project_category"])

    print(f"=== New negotiation | Archetype: {ARCHETYPE_DISPLAY_NAMES.get(info['archetype'], info['archetype'])} "
          f"| Category: {info['project_category']} | Floor: ${info['floor']:g} | Target: ${info['target']:g} ===")
    print(f"Project: {project_description}\n")

    prev_offer = info["current_offer"]
    done = False

    while not done:
        action_id, _ = model.predict(obs, deterministic=True)
        action_id = int(action_id)

        ctx = DealContext(
            freelancer_floor=info["floor"],
            freelancer_target=info["target"],
            current_offer=info["current_offer"],
            turn_number=info["turn"] + 1,
            turns_remaining=max_turns - info["turn"],
            archetype=ARCHETYPE_DISPLAY_NAMES.get(info["archetype"], info["archetype"]),
            client_last_message=_describe_client_offer(info["current_offer"], prev_offer),
            detected_sentiment=_sentiment_label(float(obs[12])),  # index 12 = sentiment_score
            project_description=project_description,
            extra_notes=f"Turn {info['turn'] + 1} of {max_turns}. "
                        f"Relationship score: {info['relationship_score']}.",
        )

        script = generator.generate_single(action_id, ctx)
        print(f"[Turn {ctx.turn_number}] Agent action: {ACTIONS[action_id]}")
        print(f"Freelancer says: {script}\n")

        prev_offer = info["current_offer"]
        obs, reward, terminated, truncated, info = env.step(action_id)
        done = terminated or truncated

    final_deal = info.get("final_deal")
    print("=" * 70)
    if final_deal:
        print(f"Deal closed at ${final_deal:g} "
              f"(target was ${info['target']:g}, floor was ${info['floor']:g})")
    else:
        print("No deal reached — negotiation ended without agreement.")
    print(f"Final episode reward: {reward}")


def run_standalone_demo():
    """Runs a batch of manually-scripted scenarios (different archetypes,
    actions, and deal states) so you can spot patterns in output quality
    across several turns instead of just one — before touching the prompt."""
    generator = ScriptGenerator()

    scenarios = [
        # Lowballer, re-anchoring up off a low opening offer.
        dict(
            ctx=DealContext(
                freelancer_floor=1800,
                freelancer_target=2500,
                current_offer=1200,
                turn_number=2,
                turns_remaining=6,
                archetype="Lowballer",
                client_last_message="Honestly 1200 is generous for this, most devs would take it.",
                detected_sentiment="dismissive",
                project_description="a 3-week backend API build for a scheduling app",
                extra_notes="Client opened low and hasn't moved yet.",
            ),
            action_id=1,  # Re-anchor Higher
        ),
        # Friendly Crusher, partial concession with a condition attached.
        dict(
            ctx=DealContext(
                freelancer_floor=1800,
                freelancer_target=2500,
                current_offer=2100,
                turn_number=4,
                turns_remaining=3,
                archetype="Friendly Crusher",
                client_last_message="I really want to work with you, can you meet me at 2100? I know you'll love this project.",
                detected_sentiment="collaborative",
                project_description="a 3-week backend API build for a scheduling app",
                extra_notes="Client has been warm throughout, this is their 2nd offer.",
            ),
            action_id=2,  # Concede Partially
        ),
        # Deadline Rusher, holding firm and reframing instead of caving to urgency.
        dict(
            ctx=DealContext(
                freelancer_floor=1800,
                freelancer_target=2500,
                current_offer=1800,
                turn_number=3,
                turns_remaining=4,
                archetype="Deadline Rusher",
                client_last_message="I need this live by Friday, can you just take 1800 so we can move fast?",
                detected_sentiment="urgent",
                project_description="a 3-week backend API build for a scheduling app",
                extra_notes="Client is pushing urgency to justify a lower number.",
            ),
            action_id=0,  # Hold and Reframe
        ),
        # Scope Creeper, setting a firm boundary on an out-of-scope ask.
        dict(
            ctx=DealContext(
                freelancer_floor=1800,
                freelancer_target=2500,
                current_offer=2000,
                turn_number=5,
                turns_remaining=2,
                archetype="Scope Creeper",
                client_last_message="Since we're at 2000, can you also add the admin dashboard? Shouldn't take long.",
                detected_sentiment="noncommittal",
                project_description="a 3-week backend API build for a scheduling app",
                extra_notes="Admin dashboard was never part of the original scope.",
            ),
            action_id=4,  # Set Boundary / Firm No
        ),
        # Ghoster reappearing after silence, agent decides to walk.
        dict(
            ctx=DealContext(
                freelancer_floor=1800,
                freelancer_target=2500,
                current_offer=1500,
                turn_number=7,
                turns_remaining=1,
                archetype="Ghoster",
                client_last_message="sorry been swamped, still interested, can we do 1500?",
                detected_sentiment="hesitant",
                project_description="a 3-week backend API build for a scheduling app",
                extra_notes="Client went silent for 5 days mid-negotiation, now lowballing on return.",
            ),
            action_id=5,  # Walk Away
        ),
    ]

    for scenario in scenarios:
        ctx = scenario["ctx"]
        action_id = scenario["action_id"]

        print(f"--- Turn {ctx.turn_number} | Archetype: {ctx.archetype} | Action: {action_id} "
              f"| Floor/Target: {ctx.freelancer_floor}/{ctx.freelancer_target} ---")
        print(f"Client said: \"{ctx.client_last_message}\"\n")

        scripts = generator.generate(action_id, ctx, n_variants=2)
        for i, script in enumerate(scripts, 1):
            print(f"Variant {i}:\n{script}\n")

        print("=" * 70 + "\n")


if __name__ == "__main__":
    if USE_TRAINED_AGENT:
        run_full_pipeline_demo()
    else:
        run_standalone_demo()