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

TASK_TO_ACTION_TYPE: dict[str, str] = {
    "task_1_classify":   "classify",
    "task_2_prioritize": "prioritize",
    "task_3_diagnose":   "diagnose",
}
STEP_PENALTY = 0.01
CONFIDENCE_BONUS = 0.1
ALWAYS_FULL_CONFIDENCE_PENALTY = 0.2
REPETITION_WINDOW = 3

def _clamp_score(x: float) -> float:
    return max(0.01, min(0.99, float(x)))

def _action_fingerprint(action: Action) -> str:
    return json.dumps(action.model_dump(), sort_keys=True)

class ManufacturingDefectEnv:
    VALID_TASK_IDS = set(TASK_TO_ACTION_TYPE.keys())

    def __init__(self, task_id: str = "task_1_classify", seed: int = 42) -> None:
        if task_id not in self.VALID_TASK_IDS:
            raise ValueError(f"task_id must be one of {self.VALID_TASK_IDS}, got '{task_id}'")
        self.task_id = task_id
        self.seed = seed
        self._step_count: int = 0
        self._done: bool = False
        self._current_score: float = 0.01
        self._scenario: Any = None
        self._current_observation: Observation | None = None
        self._action_history: list[str] = []
        self._confidence_history: list[float] = []

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

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is already done. Call reset() before stepping again.")
        if self._current_observation is None:
            raise RuntimeError("Call reset() before step().")
        self._step_count += 1
        info: dict[str, Any] = {
            "step": self._step_count,
            "task_id": self.task_id,
            "action_history_length": len(self._action_history),
        }
        expected_action_type = TASK_TO_ACTION_TYPE[self.task_id]
        if action.action_type != expected_action_type:
            self._done = True
            self._current_score = 0.01
            info["hack_detected"] = "wrong_action_type"
            return self._current_observation, Reward(
                score=_clamp_score(0.01), partial_credit=_clamp_score(0.01),
                feedback=f"Wrong action_type: expected '{expected_action_type}', got '{action.action_type}'.",
                done=True,
            ), True, info
        fingerprint = _action_fingerprint(action)
        self._action_history.append(fingerprint)
        if len(self._action_history) >= REPETITION_WINDOW and all(
            fp == self._action_history[-1] for fp in self._action_history[-REPETITION_WINDOW:]
        ):
            self._done = True
            self._current_score = 0.01
            info["hack_detected"] = "repetitive_action"
            return self._current_observation, Reward(
                score=_clamp_score(0.01), partial_credit=_clamp_score(0.01),
                feedback="Repetitive action detected — possible reward hacking.",
                done=True,
            ), True, info
        raw_credit, grader_feedback = self._grade(action)
        partial_credit = _clamp_score(raw_credit)
        score = _clamp_score(partial_credit - STEP_PENALTY * self._step_count)
        self._confidence_history.append(action.confidence)
        if len(self._confidence_history) >= 3 and all(c == 1.0 for c in self._confidence_history):
            score = _clamp_score(score - ALWAYS_FULL_CONFIDENCE_PENALTY)
            grader_feedback += " | Overconfidence penalty applied."
            info["hack_detected"] = "always_full_confidence"
        if action.confidence > 0.8 and partial_credit >= 0.94:
            score = _clamp_score(score + CONFIDENCE_BONUS)
            grader_feedback += " | Confidence bonus applied (+0.1)."
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
        info["step_penalty_applied"] = round(STEP_PENALTY * self._step_count, 4)
        return self._current_observation, reward, True, info

    def state(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_count": self._step_count,
            "current_score": self._current_score,
            "done": self._done,
            "seed": self.seed,
            "action_history": list(self._action_history),
        }

    def render(self) -> None:
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

    def _build_observation_task1(self, scenario: dict) -> Observation:
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
        focal = defects[0]
        queue = [
            {"defect_id": d["defect_id"], "machine_id": d["machine_id"],
             "shift": d["shift"], "severity_score": d["defect_image_metadata"]["severity_score"]}
            for d in defects
        ]
        return Observation(
            defect_id=focal["defect_id"], defect_description=focal["defect_description"],
            sensor_readings=focal["sensor_readings"], defect_image_metadata=focal["defect_image_metadata"],
            machine_id=focal["machine_id"], shift=focal["shift"], batch_id=focal["batch_id"],
            timestamp=focal["timestamp"], defect_queue=queue, current_task="prioritize",
        )

    def _build_observation_task3(self, scenario: dict) -> Observation:
        defects = scenario["defects"]
        focal = defects[0]
        queue = [
            {"defect_id": d["defect_id"], "machine_id": d["machine_id"],
             "shift": d["shift"], "sensor_readings": d["sensor_readings"]}
            for d in defects
        ]
        return Observation(
            defect_id=focal["defect_id"], defect_description=focal["defect_description"],
            sensor_readings=focal["sensor_readings"], defect_image_metadata=focal["defect_image_metadata"],
            machine_id=focal["machine_id"], shift=focal["shift"], batch_id=focal["batch_id"],
            timestamp=focal["timestamp"], defect_queue=queue, current_task="diagnose",
        )

    def _grade(self, action: Action) -> tuple[float, str]:
        if self.task_id == "task_1_classify":
            return classify_grader(action, self._scenario["true_category"])
        elif self.task_id == "task_2_prioritize":
            return prioritize_grader(action, get_ground_truth_priority_order(self._scenario))
        else:
            return diagnose_grader(action, self._scenario["true_root_cause"])