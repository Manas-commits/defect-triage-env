"""
env/graders.py

Deterministic graders for the Manufacturing Defect Triage RL environment.
Each grader returns (score: float, feedback: str).

All scores are strictly within (0.01, 0.99) as required by the competition.
All graders are fully deterministic: identical inputs always produce
identical outputs (no randomness involved).
"""

from __future__ import annotations

from scipy import stats

from env.models import Action

# ---------------------------------------------------------------------------
# Score bounds — competition requires strictly between 0 and 1
# ---------------------------------------------------------------------------

_MIN_SCORE = 0.01
_MAX_SCORE = 0.99


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0.01, 0.99)."""
    return max(_MIN_SCORE, min(_MAX_SCORE, score))


# ---------------------------------------------------------------------------
# Defect category families (for partial credit in Task 1)
# ---------------------------------------------------------------------------

_CATEGORY_FAMILIES: dict[str, str] = {
    "dimensional": "structural",
    "assembly":    "structural",
    "surface":     "surface",
    "cosmetic":    "surface",
    "material":    "material",
}

# ---------------------------------------------------------------------------
# Root-cause categories (for partial credit in Task 3)
# ---------------------------------------------------------------------------

_ROOT_CAUSE_CATEGORIES: dict[str, str] = {
    "tool_wear":          "machine_issues",
    "calibration_drift":  "machine_issues",
    "machine_vibration":  "machine_issues",
    "operator_error":     "human_issues",
    "material_defect":    "supply_issues",
}


# ---------------------------------------------------------------------------
# Grader 1 — Defect Classification
# ---------------------------------------------------------------------------

def classify_grader(action: Action, ground_truth: str) -> tuple[float, str]:
    """
    Grade a 'classify' action.

    Scoring (before clamping):
        0.95 — exact match with ground-truth category
        0.50 — same family (structural: dimensional+assembly,
                             surface: surface+cosmetic,
                             material: material)
        0.05 — wrong category

    All scores clamped to (0.01, 0.99).
    """
    agent_cat = action.defect_category

    if agent_cat is None:
        return _clamp(0.05), "No defect_category provided for a classify action."

    if agent_cat == ground_truth:
        return _clamp(0.95), f"Correct classification: '{agent_cat}'."

    agent_family = _CATEGORY_FAMILIES.get(agent_cat)
    truth_family = _CATEGORY_FAMILIES.get(ground_truth)

    if agent_family is not None and agent_family == truth_family:
        return (
            _clamp(0.50),
            f"Partially correct: '{agent_cat}' and '{ground_truth}' are both "
            f"in the '{agent_family}' family.",
        )

    return (
        _clamp(0.05),
        f"Incorrect classification: predicted '{agent_cat}', "
        f"ground truth was '{ground_truth}'.",
    )


# ---------------------------------------------------------------------------
# Grader 2 — Defect Queue Prioritization
# ---------------------------------------------------------------------------

def prioritize_grader(
    action: Action,
    ground_truth_order: list[str],
) -> tuple[float, str]:
    """
    Grade a 'prioritize' action using Kendall Tau rank correlation.

    Scoring:
        raw = (kendall_tau + 1) / 2   →  mapped to [0, 1]
        then scaled to [0.05, 0.95] and clamped to (0.01, 0.99)
    """
    agent_order = action.priority_order

    if not agent_order:
        return _clamp(0.05), "No priority_order provided for a prioritize action."

    truth_rank = {defect_id: i for i, defect_id in enumerate(ground_truth_order)}
    common_ids = [did for did in agent_order if did in truth_rank]

    if len(common_ids) < 2:
        return _clamp(0.05), "Not enough valid defect IDs in priority_order to compute rank correlation."

    agent_ranks = [agent_order.index(did) for did in common_ids]
    truth_ranks = [truth_rank[did] for did in common_ids]

    tau, p_value = stats.kendalltau(agent_ranks, truth_ranks)

    # Normalise Kendall Tau from [-1, 1] to [0, 1], then scale to [0.05, 0.95]
    raw = (tau + 1.0) / 2.0
    score = round(0.05 + raw * 0.90, 4)

    if score >= 0.85:
        quality = "Excellent ranking"
    elif score >= 0.65:
        quality = "Good ranking"
    elif score >= 0.45:
        quality = "Moderate ranking"
    else:
        quality = "Poor ranking"

    feedback = (
        f"{quality}: Kendall Tau = {tau:.4f}, normalised score = {score:.4f} "
        f"(p={p_value:.4f}, n={len(common_ids)} items)."
    )
    return _clamp(score), feedback


# ---------------------------------------------------------------------------
# Grader 3 — Root Cause Diagnosis
# ---------------------------------------------------------------------------

def diagnose_grader(action: Action, ground_truth_cause: str) -> tuple[float, str]:
    """
    Grade a 'diagnose' action.

    Scoring (before clamping):
        0.95 — exact match with true root cause
        0.40 — same category (machine_issues, human_issues, supply_issues)
        0.05 — wrong

    All scores clamped to (0.01, 0.99).
    """
    agent_cause = action.root_cause

    if agent_cause is None:
        return _clamp(0.05), "No root_cause provided for a diagnose action."

    if agent_cause == ground_truth_cause:
        return _clamp(0.95), f"Correct diagnosis: '{agent_cause}'."

    agent_category = _ROOT_CAUSE_CATEGORIES.get(agent_cause)
    truth_category = _ROOT_CAUSE_CATEGORIES.get(ground_truth_cause)

    if agent_category is not None and agent_category == truth_category:
        return (
            _clamp(0.40),
            f"Partially correct: '{agent_cause}' and '{ground_truth_cause}' are both "
            f"in the '{agent_category}' category.",
        )

    return (
        _clamp(0.05),
        f"Incorrect diagnosis: predicted '{agent_cause}', "
        f"ground truth was '{ground_truth_cause}'.",
    )