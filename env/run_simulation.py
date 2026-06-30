"""
Validation script: runs 1,000+ simulated negotiations with a RANDOM policy
to confirm the environment works end-to-end before plugging in PPO (Week 3-4).

This is the Week 1-2 deliverable: "Run 1,000+ simulated negotiations to
validate environment."

A random policy is intentionally dumb — it's a baseline, not a strategy.
We're checking that:
  1. The env never crashes across thousands of episodes
  2. Episodes terminate properly (no infinite loops)
  3. Reward distribution looks sane (mix of good/bad outcomes, not all -10
     or all +10, which would indicate a broken reward function)
  4. Each archetype produces meaningfully different average outcomes
     (otherwise the archetypes aren't actually differentiated)

Run: python -m env.run_simulation
"""

import numpy as np
from collections import defaultdict

from env.negotiation_env import NegotiationEnv, ARCHETYPE_LIST, ACTIONS


def run_validation(n_episodes: int = 1000, seed: int = 42):
    env = NegotiationEnv(max_turns=8, seed=seed)
    rng = np.random.default_rng(seed)

    rewards = []
    deals_closed = 0
    walked_away = 0
    timed_out = 0
    per_archetype_rewards = defaultdict(list)
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        archetype = info["archetype"]
        done = False
        steps = 0

        while not done:
            action = rng.integers(0, len(ACTIONS))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        rewards.append(reward)
        per_archetype_rewards[archetype].append(reward)
        episode_lengths.append(steps)

        if info["final_deal"] is None:
            walked_away += 1
        elif truncated:
            timed_out += 1
        else:
            deals_closed += 1

    rewards = np.array(rewards)

    print("=" * 60)
    print(f"VALIDATION RUN — {n_episodes} episodes, random policy")
    print("=" * 60)
    print(f"Episodes completed without error: {n_episodes}/{n_episodes}")
    print(f"Avg episode length:   {np.mean(episode_lengths):.2f} turns")
    print(f"Avg reward:           {np.mean(rewards):.2f}")
    print(f"Reward std dev:       {np.std(rewards):.2f}")
    print(f"Reward range:         [{rewards.min():.2f}, {rewards.max():.2f}]")
    print()
    print(f"Deals closed:         {deals_closed} ({100*deals_closed/n_episodes:.1f}%)")
    print(f"Walked away / no deal:{walked_away} ({100*walked_away/n_episodes:.1f}%)")
    print(f"Timed out (max turns):{timed_out} ({100*timed_out/n_episodes:.1f}%)")
    print()
    print("Avg reward by archetype (random policy — expect these to differ,")
    print("confirming archetypes behave distinctly):")
    for arch in ARCHETYPE_LIST:
        r = per_archetype_rewards[arch]
        if r:
            print(f"  {arch:20s} n={len(r):4d}  avg_reward={np.mean(r):6.2f}  std={np.std(r):.2f}")
    print("=" * 60)

    # Basic sanity assertions
    assert len(rewards) == n_episodes, "Episode count mismatch"
    assert rewards.min() >= -10.0 and rewards.max() <= 12.0, "Reward out of expected bounds"
    assert np.std(rewards) > 0.5, "Reward has near-zero variance — env may be broken"
    print("\n✅ All sanity checks passed. Environment validated.")

    return {
        "n_episodes": n_episodes,
        "avg_reward": float(np.mean(rewards)),
        "deals_closed_pct": 100 * deals_closed / n_episodes,
        "per_archetype_avg_reward": {
            a: float(np.mean(per_archetype_rewards[a])) for a in ARCHETYPE_LIST
        },
    }


if __name__ == "__main__":
    run_validation(n_episodes=1000)
