"""
Evaluate the trained PPO policy:
  - Win rate per archetype (deal closed at/above target)
  - Average reward / deal value per archetype
  - Benchmark vs. the random-policy baseline from Week 1-2
    (target: PPO beats random by 40%+ in average deal value)

Run: python -m agent.evaluate
"""

import numpy as np
from collections import defaultdict
from stable_baselines3 import PPO

from env.negotiation_env import NegotiationEnv, ARCHETYPE_LIST, ACTIONS
from agent.train import POLICY_PATH


def run_episodes(n_episodes, policy=None, seed=0):
    """
    policy=None runs a random policy (the Week 1-2 baseline).

    Uses its own freshly-seeded env so that random and PPO runs see the
    EXACT SAME sequence of negotiation scenarios (same floor/target/market/
    archetype draws per episode index) — otherwise comparing two runs off
    a shared, sequentially-advancing RNG stream compares different,
    unrelated negotiations.
    """
    env = NegotiationEnv(max_turns=8, seed=seed)
    rng = np.random.default_rng(seed + 1)  # separate stream for action sampling only
    rewards = []
    deal_qualities = []  # normalized (final_deal - floor) / (target - floor), comparable across episodes
    wins = defaultdict(int)
    counts = defaultdict(int)
    per_archetype_rewards = defaultdict(list)

    for episode_idx in range(n_episodes):
        # Re-seed per episode index (not once for the whole run) so the
        # scenario sequence is identical between random and PPO runs
        # regardless of how many RNG calls each policy's actions triggered
        # in earlier episodes.
        obs, info = env.reset(seed=seed + episode_idx)
        archetype = info["archetype"]
        target = info["target"]
        floor = info["floor"]
        done = False

        while not done:
            if policy is None:
                action = rng.integers(0, len(ACTIONS))
            else:
                action, _ = policy.predict(obs, deterministic=True)
                action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        rewards.append(reward)
        per_archetype_rewards[archetype].append(reward)
        counts[archetype] += 1

        final_deal = info.get("final_deal")
        if final_deal is not None:
            quality = (final_deal - floor) / max(target - floor, 1e-6)
            deal_qualities.append(quality)
            if final_deal >= target:
                wins[archetype] += 1

    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_deal_quality": float(np.mean(deal_qualities)) if deal_qualities else 0.0,
        "win_rate_overall": sum(wins.values()) / n_episodes,
        "win_rate_per_archetype": {
            a: wins[a] / counts[a] if counts[a] else 0.0 for a in ARCHETYPE_LIST
        },
        "avg_reward_per_archetype": {
            a: float(np.mean(per_archetype_rewards[a])) if per_archetype_rewards[a] else 0.0
            for a in ARCHETYPE_LIST
        },
    }


def evaluate(n_episodes: int = 1000, seed: int = 123):
    print("Loading trained policy from", POLICY_PATH + ".zip")
    model = PPO.load(POLICY_PATH)

    print(f"\nRunning {n_episodes} episodes — RANDOM baseline...")
    random_results = run_episodes(n_episodes, policy=None, seed=seed)

    print(f"Running {n_episodes} episodes — TRAINED PPO policy...")
    ppo_results = run_episodes(n_episodes, policy=model, seed=seed)

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS (same seeded scenario sequence for both)")
    print("=" * 65)
    print(f"{'Metric':30s} {'Random':>15s} {'PPO':>15s}")
    print(f"{'Avg reward':30s} {random_results['avg_reward']:15.2f} {ppo_results['avg_reward']:15.2f}")
    print(f"{'Avg deal quality (0=floor,1=target)':30s} {random_results['avg_deal_quality']:15.3f} {ppo_results['avg_deal_quality']:15.3f}")
    print(f"{'Overall win rate':30s} {random_results['win_rate_overall']*100:14.1f}% {ppo_results['win_rate_overall']*100:14.1f}%")
    print()
    print("Win rate by archetype:")
    print(f"  {'Archetype':20s} {'Random':>12s} {'PPO':>12s}")
    for a in ARCHETYPE_LIST:
        r = random_results["win_rate_per_archetype"][a] * 100
        p = ppo_results["win_rate_per_archetype"][a] * 100
        print(f"  {a:20s} {r:11.1f}% {p:11.1f}%")

    # The blueprint's success benchmark: PPO beats random by 40%+ in deal value.
    # Measured here via normalized deal quality (comparable across episodes
    # with different floor/target ranges), not raw currency.
    if random_results["avg_deal_quality"] > 0:
        lift_pct = (
            (ppo_results["avg_deal_quality"] - random_results["avg_deal_quality"])
            / abs(random_results["avg_deal_quality"])
            * 100
        )
    else:
        lift_pct = float("inf") if ppo_results["avg_deal_quality"] > 0 else 0.0

    print("\n" + "=" * 65)
    print(f"DEAL VALUE LIFT vs random: {lift_pct:+.1f}%  (target: +40% or more)")
    if lift_pct >= 40:
        print("✅ BENCHMARK MET — PPO beats random by 40%+ in deal value.")
    else:
        print("⚠️  Benchmark not yet met. Consider: more training timesteps,")
        print("    reward shaping adjustments, or hyperparameter tuning.")
    print("=" * 65)

    return {"random": random_results, "ppo": ppo_results, "deal_value_lift_pct": lift_pct}


if __name__ == "__main__":
    evaluate(n_episodes=1000)
