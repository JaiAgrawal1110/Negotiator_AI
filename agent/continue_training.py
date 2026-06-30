"""
Resume training an existing PPO checkpoint instead of starting from scratch.

Use this whenever you come back to push the agent further — e.g. more
timesteps, adjusted exploration (ent_coef), or after fixing the
Lowballer/Scope Creeper underperformance noted in the Week 3-4 eval.

reset_num_timesteps=False tells SB3 this is a continuation: it keeps the
existing policy + optimizer state (loaded from the .zip) and keeps the
internal step counter going instead of resetting logging/schedules.

Run: python -m agent.continue_training
     python -m agent.continue_training --timesteps 500000 --ent-coef 0.05
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env.negotiation_env import NegotiationEnv
from agent.train import POLICY_PATH, make_env


def continue_training(
    additional_timesteps: int = 300_000,
    n_envs: int = 4,
    seed: int = 7,
    ent_coef: float = None,
):
    print(f"Loading existing policy from {POLICY_PATH}.zip ...")
    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=seed)

    model = PPO.load(POLICY_PATH, env=vec_env)

    if ent_coef is not None:
        print(f"Overriding ent_coef -> {ent_coef} (raise this to push more "
              f"exploration into underperforming archetypes like Lowballer "
              f"/ Scope Creeper)")
        model.ent_coef = ent_coef

    print(f"Resuming training for {additional_timesteps} more timesteps "
          f"(continuing from prior checkpoint, not restarting)...")
    model.learn(
        total_timesteps=additional_timesteps,
        reset_num_timesteps=False,
        progress_bar=False,
    )

    model.save(POLICY_PATH)
    print(f"Updated policy saved to {POLICY_PATH}.zip")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--ent-coef", type=float, default=None,
                         help="Raise this (e.g. 0.05) to push more exploration "
                              "into underrepresented archetypes.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    continue_training(
        additional_timesteps=args.timesteps,
        seed=args.seed,
        ent_coef=args.ent_coef,
    )
