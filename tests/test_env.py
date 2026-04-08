"""
tests/test_env.py

pytest test suite for the Manufacturing Defect Triage RL environment.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import pytest

from env.environment import ManufacturingDefectEnv
from env.graders import classify_grader, diagnose_grader, prioritize_grader
from env.models import Action, Observation, Reward
from env.tasks import (
    get_task_1_scenario,
    get_task_2_scenario,
    get_task_3_scenario,
    get_ground_truth_priority_order,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_classify_action(category: str, confidence: float = 0.9) -> Action:
    return Action(action_type="classify", defect_category=category, confidence=confidence)


def _make_prioritize_action(order: list[str], confidence: float = 0.9) -> Action:
    return Action(action_type="prioritize", priority_order=order, confidence=confidence)


def _make_diagnose_action(root_cause: str, confidence: float = 0.9) -> Action:
    return Action(action_type="diagnose", root_cause=root_cause, confidence=confidence)


# ---------------------------------------------------------------------------
# Test 1 — reset returns a valid Observation
# ---------------------------------------------------------------------------

def test_reset_returns_observation():
    """reset() must return a valid Observation Pydantic model."""
    env = ManufacturingDefectEnv(task_id="task_1_classify", seed=42)
    obs = env.reset()

    assert isinstance(obs, Observation)
    assert obs.defect_id.startswith("DEF-")
    assert obs.current_task == "classify"
    assert isinstance(obs.sensor_readings, dict)
    assert "temperature" in obs.sensor_readings
    assert "vibration" in obs.sensor_readings
    assert "pressure" in obs.sensor_readings


# ---------------------------------------------------------------------------
# Test 2 — step returns correct types
# ---------------------------------------------------------------------------

def test_step_returns_correct_types():
    """step() must return (Observation, Reward, bool, dict)."""
    env = ManufacturingDefectEnv(task_id="task_1_classify", seed=42)
    env.reset()

    action = _make_classify_action("surface")
    result = env.step(action)

    assert len(result) == 4
    obs, reward, done, info = result
    assert isinstance(obs, Observation)
    assert isinstance(reward, Reward)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert done is True  # single-step episode always ends


# ---------------------------------------------------------------------------
# Test 3 — correct classification scores 1.0 (partial credit)
# ---------------------------------------------------------------------------

def test_grader_task1_correct():
    """A correct classification must return partial_credit = 1.0."""
    scenario = get_task_1_scenario(seed=42)
    true_category = scenario["true_category"]

    action = _make_classify_action(true_category)
    score, feedback = classify_grader(action, true_category)

    assert score == 1.0, f"Expected 1.0, got {score}. Feedback: {feedback}"
    assert "Correct" in feedback


# ---------------------------------------------------------------------------
# Test 4 — wrong classification scores 0.0 (no family match)
# ---------------------------------------------------------------------------

def test_grader_task1_wrong():
    """A wrong classification with no family overlap must return score 0.0."""
    # 'material' has no family partner — anything else in a different family
    action = _make_classify_action("material")
    score, feedback = classify_grader(action, "surface")

    assert score == 0.0, f"Expected 0.0, got {score}. Feedback: {feedback}"
    assert "Incorrect" in feedback


# ---------------------------------------------------------------------------
# Test 5 — perfect prioritization scores 1.0
# ---------------------------------------------------------------------------

def test_grader_task2_perfect_order():
    """A perfectly ordered priority list must return score 1.0."""
    defects = get_task_2_scenario(seed=42)
    ground_truth_order = get_ground_truth_priority_order(defects)

    action = _make_prioritize_action(ground_truth_order)
    score, feedback = prioritize_grader(action, ground_truth_order)

    assert score == 1.0, f"Expected 1.0, got {score}. Feedback: {feedback}"


# ---------------------------------------------------------------------------
# Test 6 — exact root cause match scores 1.0
# ---------------------------------------------------------------------------

def test_grader_task3_exact_match():
    """An exact root cause prediction must return score 1.0."""
    scenario = get_task_3_scenario(seed=42)
    true_cause = scenario["true_root_cause"]

    action = _make_diagnose_action(true_cause)
    score, feedback = diagnose_grader(action, true_cause)

    assert score == 1.0, f"Expected 1.0, got {score}. Feedback: {feedback}"
    assert "Correct" in feedback


# ---------------------------------------------------------------------------
# Test 7 — reproducibility: same seed → same first observation
# ---------------------------------------------------------------------------

def test_reproducibility():
    """Two resets with the same seed must produce identical observations."""
    env = ManufacturingDefectEnv(task_id="task_1_classify", seed=42)

    obs_a = env.reset()
    obs_b = env.reset()

    assert obs_a.defect_id == obs_b.defect_id
    assert obs_a.defect_description == obs_b.defect_description
    assert obs_a.sensor_readings == obs_b.sensor_readings
    assert obs_a.defect_image_metadata == obs_b.defect_image_metadata


# ---------------------------------------------------------------------------
# Test 8 — state() returns dict with expected keys
# ---------------------------------------------------------------------------

def test_state_returns_dict():
    """state() must return a dict containing the required keys."""
    env = ManufacturingDefectEnv(task_id="task_2_prioritize", seed=42)
    env.reset()

    s = env.state()
    required_keys = {"task_id", "step_count", "current_score", "done", "seed"}

    assert isinstance(s, dict)
    for key in required_keys:
        assert key in s, f"Missing key '{key}' in state() output"

    assert s["task_id"] == "task_2_prioritize"
    assert s["seed"] == 42
    assert isinstance(s["step_count"], int)
    assert isinstance(s["done"], bool)