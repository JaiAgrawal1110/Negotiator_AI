"""
Gymnasium custom environment for negotiation simulation.

Design note: RL training needs thousands of fast episodes. Calling an LLM
for every single step would be slow and burn API quota. So this env uses
a lightweight RULE-BASED price-transition model (directly encoding each
archetype's win/walk conditions from archetypes.py) to drive state
transitions during training. The LLM (llm/simulator.py, Groq) is used
separately to generate the actual human-readable dialogue for demos and
for the few-shot archetype classifier — not inside the hot training loop.

This mirrors a common RL pattern: train fast on a structured simulator,
then validate qualitatively against the LLM-driven version before going
live with real users.

STATE SPACE (Box, 13 floats, all normalized to roughly [0, 1] or [-1, 1]):
    0  price_position      -> (current_offer - floor) / (target - floor), clipped [0,1]
    1  turns_taken_norm     -> turns_taken / max_turns
    2  turns_remaining_norm -> (max_turns - turns_taken) / max_turns
    3-7 archetype_onehot    -> 5-dim one-hot: [lowballer, ghoster, friendly_crusher,
                                                 deadline_rusher, scope_creeper]
    8  leverage_score       -> 0-1, freelancer's leverage (competing offers, portfolio, etc.)
    9  relationship_score   -> 0-1, how warm/damaged the relationship is, starts at 0.7
    10 deadline_urgency     -> 0-1, how rushed the client is
    11 last_offer_delta     -> normalized change in offer from previous turn [-1,1]
    12 sentiment_score      -> 0-1 placeholder; wired to real BERT output in Week 5-6
                               (nlp/sentiment.py). Defaults to a neutral 0.5 for now.

ACTION SPACE (Discrete(6)) — the 6 negotiation strategies the RL agent picks from:
    0  HOLD_AND_REFRAME   -> restate value, don't move price
    1  RE_ANCHOR_HIGHER   -> counter with a higher number than current ask
    2  CONCEDE_PARTIAL    -> move price down partway toward client's ask
    3  ADD_VALUE          -> offer non-price value (timeline, scope clarity) instead of moving price
    4  SET_BOUNDARY       -> firm no / name a hard floor or scope limit
    5  WALK_AWAY          -> end the negotiation (used when deal is below floor and client won't move)

REWARD FUNCTION: ports calculate_reward() from the blueprint exactly
(section 05), applied once per episode at termination.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

ARCHETYPE_LIST = [
    "lowballer",
    "ghoster",
    "friendly_crusher",
    "deadline_rusher",
    "scope_creeper",
]

# Same 8 categories as the real Kaggle market-rate dataset (Week 9-10),
# so every episode's project_category lines up with something the market
# retriever actually has data for. Kept as a plain constant (not read from
# a CSV) so reset() stays a cheap, I/O-free operation -- this env still
# needs to run thousands of episodes fast for RL training, per the module
# docstring's design philosophy. Only a category LABEL is assigned here;
# turning that into an actual project description (for retrieval) happens
# one layer up, in scripts/demo_negotiation.py, not inside the env itself.
PROJECT_CATEGORIES = [
    "App Development",
    "Content Writing",
    "Customer Support",
    "Data Entry",
    "Digital Marketing",
    "Graphic Design",
    "SEO",
    "Web Development",
]

ACTIONS = [
    "hold_and_reframe",
    "re_anchor_higher",
    "concede_partial",
    "add_value",
    "set_boundary",
    "walk_away",
]


def calculate_reward(final_deal, target, floor, market_rate, relationship_score):
    """
    Tuned version of the blueprint's reward function (section 05).

    TUNING NOTE (Week 3-4): the original version scored the floor-to-target
    range LINEARLY. First training run showed the agent exploiting this —
    it learned to reliably bank "safe," mediocre deals around the middle of
    the range instead of pushing for target, because a 50%-of-the-way deal
    scored a proportionally decent 4/8 with much lower risk than gambling
    on reaching target. Result: avg reward went up but win rate (deals at
    or above target) and avg deal value both got WORSE vs. a random
    baseline — a form of reward under-shooting, not the "agent finds an
    exploit" kind of reward hacking, but the same underlying problem: the
    reward landscape didn't actually match the business goal.

    Fix: make the floor-to-target curve CONVEX (quadratic) instead of
    linear, so middling deals are penalized much more relative to deals
    close to target. This removes the "good enough" plateau the agent was
    settling into and makes pushing closer to target clearly worth the
    risk. A small SETTLE_PENALTY is also subtracted on any below-target
    outcome so the agent isn't indifferent between "barely above floor"
    and "doing nothing."
    """
    SETTLE_PENALTY = 1.5

    if final_deal is None:
        return -10.0

    if final_deal < floor:
        return -10.0

    if final_deal >= target:
        return 10.0

    x = (final_deal - floor) / (target - floor)  # in (0, 1)
    base = (x ** 2) * 8  # convex: punishes mid-range deals much more than linear did

    if final_deal > market_rate:
        base += 2

    if relationship_score < 0.3:
        base -= 3

    base -= SETTLE_PENALTY

    return round(base, 2)


class NegotiationEnv(gym.Env):
    """
    Single-episode negotiation simulation against one client archetype.

    Episode flow:
      reset()  -> sample a deal context (archetype, target, floor, market rate, etc.)
      step(a)  -> agent picks a strategy (0-5), env resolves the client's counter-move
                  using rule-based archetype logic, returns new state + step reward 0
                  (reward is sparse, only given at episode end) + done flag.
      Episode ends when: deal is reached, client/agent walks away, or max_turns hit.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_turns: int = 8, seed: int = None):
        super().__init__()
        self.max_turns = max_turns
        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )

        # Episode state, set in reset()
        self.archetype = None
        self.project_category = None
        self.target = None
        self.floor = None
        self.market_rate = None
        self.quoted_rate = None
        self.current_offer = None  # the live number on the table
        self.turns_taken = 0
        self.leverage_score = 0.0
        self.relationship_score = 0.7
        self.deadline_urgency = 0.0
        self.last_offer_delta = 0.0
        self.sentiment_score = 0.5
        self._done = False

    # ------------------------------------------------------------------ #
    # Core Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}

        self.archetype = options.get(
            "archetype", self._rng.choice(ARCHETYPE_LIST)
        )
        self.project_category = options.get(
            "project_category", self._rng.choice(PROJECT_CATEGORIES)
        )

        # Sample a realistic deal context. floor < market_rate <= target.
        self.floor = float(options.get("floor", self._rng.uniform(1500, 6000)))
        self.target = float(
            options.get("target", self.floor * self._rng.uniform(1.15, 1.45))
        )
        self.market_rate = float(
            options.get(
                "market_rate", self._rng.uniform(self.floor, self.target)
            )
        )
        self.quoted_rate = float(
            options.get("quoted_rate", self.target * self._rng.uniform(0.95, 1.05))
        )

        # Client opens with an archetype-flavored opening offer.
        self.current_offer = self._client_opening_offer()

        self.turns_taken = 0
        self.leverage_score = float(options.get("leverage_score", self._rng.uniform(0.2, 0.9)))
        self.relationship_score = 0.7
        self.deadline_urgency = float(options.get("deadline_urgency", self._rng.uniform(0.1, 0.9)))
        self.last_offer_delta = 0.0
        self.sentiment_score = 0.5
        self._done = False

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() called after episode finished. Call reset().")

        action_name = ACTIONS[action]
        self.turns_taken += 1

        terminated = False
        truncated = False
        final_deal = None

        if action_name == "walk_away":
            terminated = True
            final_deal = None

        else:
            freelancer_offer = self._resolve_freelancer_offer(action_name)
            client_offer, deal_closed, client_walked, rel_delta = self._client_response(
                action_name, freelancer_offer
            )

            self.last_offer_delta = float(np.clip(
                (client_offer - self.current_offer) / max(self.target - self.floor, 1e-6),
                -1.0,
                1.0,
            ))
            self.current_offer = client_offer
            self.relationship_score = float(np.clip(self.relationship_score + rel_delta, 0.0, 1.0))

            if deal_closed:
                terminated = True
                final_deal = client_offer
            elif client_walked:
                terminated = True
                final_deal = None

        if self.turns_taken >= self.max_turns and not terminated:
            truncated = True
            final_deal = self.current_offer

        reward = 0.0
        if terminated or truncated:
            self._done = True
            reward = calculate_reward(
                final_deal=final_deal,
                target=self.target,
                floor=self.floor,
                market_rate=self.market_rate,
                relationship_score=self.relationship_score,
            )

        info = self._get_info()
        info["final_deal"] = final_deal

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _client_opening_offer(self) -> float:
        if self.archetype == "lowballer":
            return self.floor + (self.target - self.floor) * self._rng.uniform(-0.1, 0.15)
        if self.archetype == "deadline_rusher":
            return self.target * self._rng.uniform(0.85, 0.95)
        return self.quoted_rate * self._rng.uniform(0.85, 0.97)

    def _resolve_freelancer_offer(self, action_name: str) -> float:
        """What number the freelancer is effectively putting on the table this turn."""
        gap = self.quoted_rate - self.current_offer
        if action_name == "hold_and_reframe":
            return self.quoted_rate
        if action_name == "re_anchor_higher":
            return self.quoted_rate * 1.03
        if action_name == "concede_partial":
            return self.current_offer + gap * 0.4
        if action_name == "add_value":
            return self.quoted_rate
        if action_name == "set_boundary":
            return max(self.floor, self.quoted_rate * 0.98)
        return self.quoted_rate

    def _client_response(self, action_name: str, freelancer_offer: float):
        """
        Rule-based client counter-move, encoding each archetype's
        win/walk conditions from archetypes.py.
        Returns: (new_client_offer, deal_closed, client_walked, relationship_delta)
        """
        rng = self._rng
        gap = freelancer_offer - self.current_offer
        firm_actions = {"hold_and_reframe", "set_boundary", "re_anchor_higher"}
        soft_actions = {"concede_partial"}

        if self.archetype == "lowballer":
            if action_name in firm_actions:
                new_offer = self.current_offer + gap * rng.uniform(0.5, 0.75)
                if rng.random() < 0.2:
                    return new_offer, True, False, 0.02
                return new_offer, False, False, 0.0
            if action_name in soft_actions:
                new_offer = self.current_offer - abs(gap) * 0.2
                return max(new_offer, self.floor * 0.8), False, False, -0.05
            return self.current_offer, False, False, 0.0

        if self.archetype == "ghoster":
            if self.turns_taken >= 2 and action_name in firm_actions and rng.random() < 0.4:
                new_offer = self.current_offer + gap * 0.6
                return new_offer, rng.random() < 0.3, False, 0.05
            return self.current_offer, False, rng.random() < 0.1, -0.02

        if self.archetype == "friendly_crusher":
            if action_name in firm_actions:
                if rng.random() < 0.5:
                    return freelancer_offer, True, False, 0.05
                return self.current_offer + gap * 0.7, False, False, 0.02
            if action_name == "add_value":
                return self.current_offer, rng.random() < 0.3, False, 0.05
            new_offer = self.current_offer - abs(gap) * 0.15
            return max(new_offer, self.floor), False, False, -0.02

        if self.archetype == "deadline_rusher":
            if action_name in firm_actions or action_name == "add_value":
                new_offer = self.current_offer + gap * rng.uniform(0.6, 0.9)
                return new_offer, rng.random() < 0.45, False, 0.03
            new_offer = self.current_offer - abs(gap) * 0.1
            return new_offer, False, False, 0.0

        if self.archetype == "scope_creeper":
            if action_name == "set_boundary":
                return freelancer_offer, rng.random() < 0.4, False, 0.04
            if action_name in {"hold_and_reframe", "re_anchor_higher"}:
                new_offer = self.current_offer + gap * 0.5
                return new_offer, rng.random() < 0.25, False, 0.0
            return self.current_offer, False, False, -0.04

        return self.current_offer, False, False, 0.0

    def update_sentiment(self, score: float) -> None:
        """
        Wire a real sentiment score (0-1) into the state vector, replacing
        the 0.5 placeholder. Call this after running the client's latest
        message through nlp.sentiment.ToneClassifier.sentiment_score_for_state_vector()
        each turn, BEFORE the next step()/observation is read.

        Kept as a separate setter (rather than auto-computed inside step())
        because the env itself has no text/dialogue -- it only sees numbers.
        Real text only exists in the LLM-driven dialogue layer
        (llm/simulator.py for sim, or the live API in Week 11-12), so the
        sentiment score has to be computed there and pushed in here.

        Week 5-6: wires the NLP sentiment layer into the RL observation
        space, replacing the static 0.5 placeholder used during Week 1-2/3-4.
        """
        self.sentiment_score = float(max(0.0, min(1.0, score)))

    def _get_obs(self) -> np.ndarray:
        price_position = float(np.clip(
            (self.current_offer - self.floor) / max(self.target - self.floor, 1e-6),
            0.0,
            1.0,
        ))
        archetype_onehot = [
            1.0 if self.archetype == a else 0.0 for a in ARCHETYPE_LIST
        ]
        obs = np.array(
            [
                price_position,
                self.turns_taken / self.max_turns,
                (self.max_turns - self.turns_taken) / self.max_turns,
                *archetype_onehot,
                self.leverage_score,
                self.relationship_score,
                self.deadline_urgency,
                self.last_offer_delta,
                self.sentiment_score,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> dict:
        return {
            "archetype": self.archetype,
            "project_category": self.project_category,
            "turn": self.turns_taken,
            "current_offer": round(self.current_offer, 2),
            "target": round(self.target, 2),
            "floor": round(self.floor, 2),
            "market_rate": round(self.market_rate, 2),
            "relationship_score": round(self.relationship_score, 2),
        }