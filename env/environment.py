"""
env/environment.py
"""

from __future__ import annotations

import json
from typing import Any

from env.graders import classify_grader, diagnose_grader, prioritize_grader
from env.models import Action, Observation, Reward
from env.tasks import (
    get_ground_truth_priority_order,
    get_task_1_scenario,
    get_task_2_scenario,
    get_task_3_scenario,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_TO_ACTION_TYPE: dict[str, str] = {
    "task_1_classify":   "classify",
    "task_2_prioritize": "prioritize",
    "task_3_diagnose":   "diagnose",
}

STEP_PENALTY = 0.01
CONFIDENCE_BONUS = 0.1
ALWAYS_FULL_CONFIDENCE_PENALTY = 0.2
REPETITION_WINDOW = 3


# 🔴 GLOBAL SAFE CLAMP (NEW - CRITICAL FIX)
def _clamp_score(x: float) -> float:
    return max(0.01, min(0.99, x))


def _action_fingerprint(action: Action) -> str:
    return json.dumps(action.model_dump(), sort_keys=True)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ManufacturingDefectEnv:

    VALID_TASK_IDS = set(TASK_TO_ACTION_TYPE.keys())

    def __init__(self, task_id: str = "task_1_classify", seed: int = 42) -> None:
        if task_id not in self.VALID_TASK_IDS:
            raise ValueError(
                f"task_id must be one of {self.VALID_TASK_IDS}, got '{task_id}'"
            )
        self.task_id = task_id
        self.seed = seed

        self._step_count: int = 0
        self._done: bool = False
        self._current_score: float = 0.0
        self._scenario: Any = None
        self._current_observation: Observation | None = None
        self._action_history: list[str] = []
        self._confidence_history: list[float] = []

    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        self._step_count = 0
        self._done = False
        self._current_score = 0.01
        self._action_history = []
        self._confidence_history = []

        if self.task_id == "task_1_classify":
            self._scenario = get_task_1_scenario(seed=self.seed)
            obs = self._build_observation_task1(self._scenario)

        elif self.task_id == "task_2_prioritize":
            self._scenario = get_task_2_scenario(seed=self.seed)
            obs = self._build_observation_task2(self._scenario)

        else:
            self._scenario = get_task_3_scenario(seed=self.seed)
            obs = self._build_observation_task3(self._scenario)

        self._current_observation = obs
        return obs

    # ------------------------------------------------------------------

    def step(self, action: Action):

        if self._done:
            raise RuntimeError("Episode is already done.")
        if self._current_observation is None:
            raise RuntimeError("Call reset() before step().")

        self._step_count += 1

        info: dict[str, Any] = {
            "step": self._step_count,
            "task_id": self.task_id,
            "action_history_length": len(self._action_history),
        }

        expected_action_type = TASK_TO_ACTION_TYPE[self.task_id]

        # ❗ WRONG ACTION TYPE
        if action.action_type != expected_action_type:
            reward = Reward(
                score=0.01,
                partial_credit=0.01,
                feedback=f"Wrong action_type: expected '{expected_action_type}', got '{action.action_type}'.",
                done=True,
            )
            self._done = True
            self._current_score = 0.01
            return self._current_observation, reward, True, info

        # ❗ REPETITION CHECK
        fingerprint = _action_fingerprint(action)
        self._action_history.append(fingerprint)

        if len(self._action_history) >= REPETITION_WINDOW and all(
            fp == self._action_history[-1]
            for fp in self._action_history[-REPETITION_WINDOW:]
        ):
            reward = Reward(
                score=0.01,
                partial_credit=0.01,
                feedback="Repetitive action detected.",
                done=True,
            )
            self._done = True
            self._current_score = 0.01
            return self._current_observation, reward, True, info

        # -------------------------
        # 🔴 MAIN FIX STARTS HERE
        # -------------------------

        partial_credit, grader_feedback = self._grade(action)

        # 🔴 CRITICAL: clamp partial_credit
        partial_credit = _clamp_score(partial_credit)

        # STEP PENALTY
        score = partial_credit - STEP_PENALTY * self._step_count
        score = _clamp_score(score)

        # CONFIDENCE HISTORY
        self._confidence_history.append(action.confidence)

        # OVERCONFIDENCE PENALTY
        if (
            len(self._confidence_history) >= 3
            and all(c == 1.0 for c in self._confidence_history)
        ):
            score = _clamp_score(score - ALWAYS_FULL_CONFIDENCE_PENALTY)
            grader_feedback += " | Overconfidence penalty applied."

        # CONFIDENCE BONUS
        if action.confidence > 0.8 and partial_credit >= 0.94:
            score = _clamp_score(score + CONFIDENCE_BONUS)
            grader_feedback += " | Confidence bonus applied."

        # FINAL SAFETY CLAMP
        score = _clamp_score(score)

        self._current_score = score
        self._done = True

        reward = Reward(
            score=_clamp_score(score),
            partial_credit=_clamp_score(partial_credit),
            feedback=grader_feedback,
            done=True,
        )

        info["partial_credit"] = partial_credit
        info["step_penalty_applied"] = STEP_PENALTY * self._step_count

        return self._current_observation, reward, True, info

    # ------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_count": self._step_count,
            "current_score": self._current_score,
            "done": self._done,
            "seed": self.seed,
            "action_history": list(self._action_history),
        }

    # ------------------------------------------------------------------

    def _grade(self, action: Action):

        if self.task_id == "task_1_classify":
            return classify_grader(action, self._scenario["true_category"])

        elif self.task_id == "task_2_prioritize":
            order = get_ground_truth_priority_order(self._scenario)
            return prioritize_grader(action, order)

        else:
            return diagnose_grader(action, self._scenario["true_root_cause"])