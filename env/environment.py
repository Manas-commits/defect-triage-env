"""
env/environment.py

Main RL environment: ManufacturingDefectEnv

Implements the OpenEnv spec:
    - reset() -> Observation
    - step(action) -> tuple[Observation, Reward, bool, dict]
    - state() -> dict

Includes anti-reward-hacking checks:
    - Repetitive action detection (same action 3× in a row → score = 0.0)
    - Always-maximum-confidence penalty (confidence always 1.0 → -0.2)
    - Wrong action_type for current task → score = 0.0
"""

from __future__ import annotations

import copy
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

STEP_PENALTY = 0.01          # Deducted per step to penalise slow agents
CONFIDENCE_BONUS = 0.1       # Awarded when confidence > 0.8 AND answer is correct
ALWAYS_FULL_CONFIDENCE_PENALTY = 0.2  # Applied when agent always reports confidence = 1.0
REPETITION_WINDOW = 3        # Number of identical consecutive actions that triggers penalty


def _action_fingerprint(action: Action) -> str:
    """Stable string representation of an action for repetition detection."""
    return json.dumps(action.model_dump(), sort_keys=True)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ManufacturingDefectEnv:
    """
    OpenEnv-compliant RL environment for Manufacturing Defect Triage.

    The environment presents one of three tasks:
        - task_1_classify:    Classify a single defect into a category.
        - task_2_prioritize:  Rank a queue of 10 defects by repair priority.
        - task_3_diagnose:    Identify the root cause from a pattern of defects.

    Args:
        task_id: Which task to run (default: 'task_1_classify').
        seed:    Random seed for reproducibility (default: 42).
    """

    VALID_TASK_IDS = set(TASK_TO_ACTION_TYPE.keys())

    def __init__(self, task_id: str = "task_1_classify", seed: int = 42) -> None:
        if task_id not in self.VALID_TASK_IDS:
            raise ValueError(
                f"task_id must be one of {self.VALID_TASK_IDS}, got '{task_id}'"
            )
        self.task_id = task_id
        self.seed = seed

        # Internal state — populated on reset()
        self._step_count: int = 0
        self._done: bool = False
        self._current_score: float = 0.0
        self._scenario: Any = None          # raw scenario data (with ground truth)
        self._current_observation: Observation | None = None
        self._action_history: list[str] = []  # fingerprints for repetition detection
        self._confidence_history: list[float] = []

    # ------------------------------------------------------------------
    # OpenEnv core methods
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """
        Reset the environment to a fresh episode.

        Generates a new scenario based on task_id and seed, resets all
        internal state counters.

        Returns:
            Initial Observation for the agent.
        """
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

        else:  # task_3_diagnose
            self._scenario = get_task_3_scenario(seed=self.seed)
            obs = self._build_observation_task3(self._scenario)

        self._current_observation = obs
        return obs

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Execute one agent step.

        Args:
            action: An Action object produced by the agent.

        Returns:
            Tuple of (next_observation, reward, done, info_dict).
            - next_observation: same observation (single-step episode)
            - reward: Reward object with score, partial_credit, feedback, done
            - done: whether the episode is over
            - info_dict: auxiliary diagnostics
        """
        if self._done:
            raise RuntimeError(
                "Episode is already done. Call reset() before stepping again."
            )
        if self._current_observation is None:
            raise RuntimeError("Call reset() before step().")

        self._step_count += 1
        info: dict[str, Any] = {
            "step": self._step_count,
            "task_id": self.task_id,
            "action_history_length": len(self._action_history),
        }
        expected_action_type = TASK_TO_ACTION_TYPE[self.task_id]

        # --- Anti-reward-hacking check 1: wrong action type ---
        if action.action_type != expected_action_type:
            reward = Reward(
                score=0.01,
                partial_credit=0.01,
                feedback=(
                    f"Wrong action_type: expected '{expected_action_type}' "
                    f"for task '{self.task_id}', got '{action.action_type}'."
                ),
                done=True,
            )
            self._done = True
            self._current_score = 0.01
            info["hack_detected"] = "wrong_action_type"
            return self._current_observation, reward, True, info

        # --- Anti-reward-hacking check 2: repetitive actions ---
        fingerprint = _action_fingerprint(action)
        self._action_history.append(fingerprint)
        if len(self._action_history) >= REPETITION_WINDOW and all(
            fp == self._action_history[-1]
            for fp in self._action_history[-REPETITION_WINDOW:]
        ):
            reward = Reward(
                score=0.01,
                partial_credit=0.01,
                feedback="Repetitive action detected — possible reward hacking.",
                done=True,
            )
            self._done = True
            self._current_score = 0.01
            info["hack_detected"] = "repetitive_action"
            return self._current_observation, reward, True, info

        # --- Grade the action ---
        partial_credit, grader_feedback = self._grade(action)

        # --- Step penalty ---
        score = max(0.01, partial_credit - STEP_PENALTY * self._step_count)

        # --- Anti-reward-hacking check 3: always-1.0 confidence ---
        self._confidence_history.append(action.confidence)
        if (
            len(self._confidence_history) >= 3
            and all(c == 1.0 for c in self._confidence_history)
        ):
            score = max(0.01, score - ALWAYS_FULL_CONFIDENCE_PENALTY)
            grader_feedback += (
                " | Overconfidence penalty applied: confidence was 1.0 on every step."
            )
            info["hack_detected"] = "always_full_confidence"

        # --- Confidence bonus ---
        if action.confidence > 0.8 and partial_credit >= 0.94:
            score = min(0.99, score + CONFIDENCE_BONUS)
            grader_feedback += " | Confidence bonus applied (+0.1)."

        score = round(max(0.01, min(0.99, score)), 4)
        self._current_score = score
        self._done = True  # single-step episode per task

        reward = Reward(
            score=score,
            partial_credit=partial_credit,
            feedback=grader_feedback,
            done=True,
        )
        info["partial_credit"] = partial_credit
        info["step_penalty_applied"] = round(STEP_PENALTY * self._step_count, 4)

        return self._current_observation, reward, True, info

    def state(self) -> dict[str, Any]:
        """
        Return the current internal state as a plain dict.

        Returns:
            Dict with keys: task_id, step_count, current_score, done, seed,
            action_history (list of fingerprints).
        """
        return {
            "task_id": self.task_id,
            "step_count": self._step_count,
            "current_score": self._current_score,
            "done": self._done,
            "seed": self.seed,
            "action_history": list(self._action_history),
        }

    def render(self) -> None:
        """Print a simple text summary of the current environment state."""
        state = self.state()
        obs = self._current_observation
        print("=" * 60)
        print(f"  ManufacturingDefectEnv — {state['task_id']}")
        print("=" * 60)
        print(f"  Step:          {state['step_count']}")
        print(f"  Done:          {state['done']}")
        print(f"  Current score: {state['current_score']}")
        print(f"  Seed:          {state['seed']}")
        if obs:
            print(f"  Defect ID:     {obs.defect_id}")
            print(f"  Task:          {obs.current_task}")
            print(f"  Machine:       {obs.machine_id}  Shift: {obs.shift}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Internal helpers — Observation builders
    # ------------------------------------------------------------------

    def _build_observation_task1(self, scenario: dict) -> Observation:
        """Build Observation from a Task 1 (classify) scenario dict."""
        return Observation(
            defect_id=scenario["defect_id"],
            defect_description=scenario["defect_description"],
            sensor_readings=scenario["sensor_readings"],
            defect_image_metadata=scenario["defect_image_metadata"],
            machine_id=scenario["machine_id"],
            shift=scenario["shift"],
            batch_id=scenario["batch_id"],
            timestamp=scenario["timestamp"],
            defect_queue=[],
            current_task="classify",
        )

    def _build_observation_task2(self, defects: list[dict]) -> Observation:
        """Build Observation from Task 2 (prioritize) — first defect as focal point."""
        focal = defects[0]
        # Queue contains public fields of all defects
        queue = [
            {
                "defect_id": d["defect_id"],
                "machine_id": d["machine_id"],
                "shift": d["shift"],
                "severity_score": d["defect_image_metadata"]["severity_score"],
            }
            for d in defects
        ]
        return Observation(
            defect_id=focal["defect_id"],
            defect_description=focal["defect_description"],
            sensor_readings=focal["sensor_readings"],
            defect_image_metadata=focal["defect_image_metadata"],
            machine_id=focal["machine_id"],
            shift=focal["shift"],
            batch_id=focal["batch_id"],
            timestamp=focal["timestamp"],
            defect_queue=queue,
            current_task="prioritize",
        )

    def _build_observation_task3(self, scenario: dict) -> Observation:
        """Build Observation from Task 3 (diagnose) — pattern of defects."""
        defects = scenario["defects"]
        focal = defects[0]
        queue = [
            {
                "defect_id": d["defect_id"],
                "machine_id": d["machine_id"],
                "shift": d["shift"],
                "sensor_readings": d["sensor_readings"],
            }
            for d in defects
        ]
        return Observation(
            defect_id=focal["defect_id"],
            defect_description=focal["defect_description"],
            sensor_readings=focal["sensor_readings"],
            defect_image_metadata=focal["defect_image_metadata"],
            machine_id=focal["machine_id"],
            shift=focal["shift"],
            batch_id=focal["batch_id"],
            timestamp=focal["timestamp"],
            defect_queue=queue,
            current_task="diagnose",
        )

    # ------------------------------------------------------------------
    # Internal helpers — Grading dispatch
    # ------------------------------------------------------------------

    def _grade(self, action: Action) -> tuple[float, str]:
        """Dispatch to the appropriate grader and return (partial_credit, feedback)."""
        if self.task_id == "task_1_classify":
            ground_truth = self._scenario["true_category"]
            return classify_grader(action, ground_truth)

        elif self.task_id == "task_2_prioritize":
            ground_truth_order = get_ground_truth_priority_order(self._scenario)
            return prioritize_grader(action, ground_truth_order)

        else:  # task_3_diagnose
            ground_truth_cause = self._scenario["true_root_cause"]
            return diagnose_grader(action, ground_truth_cause)