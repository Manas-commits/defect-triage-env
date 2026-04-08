"""
env/graders.py

Deterministic graders for the Manufacturing Defect Triage RL environment.
Each grader returns (score: float, feedback: str).

All graders are fully deterministic: identical inputs always produce
identical outputs (no randomness involved).
"""

from __future__ import annotations

from scipy import stats

from env.models import Action

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

    Scoring:
        1.0  — exact match with ground-truth category
        0.5  — same family (structural: dimensional+assembly,
                             surface: surface+cosmetic,
                             material: material)
        0.0  — wrong category

    Args:
        action:       Agent's Action object (action_type must be 'classify').
        ground_truth: The true defect category string.

    Returns:
        Tuple of (score, feedback_string).
    """
    agent_cat = action.defect_category

    if agent_cat is None:
        return 0.0, "No defect_category provided for a classify action."

    if agent_cat == ground_truth:
        return 1.0, f"Correct classification: '{agent_cat}'."

    agent_family = _CATEGORY_FAMILIES.get(agent_cat)
    truth_family = _CATEGORY_FAMILIES.get(ground_truth)

    if agent_family is not None and agent_family == truth_family:
        return (
            0.5,
            f"Partially correct: '{agent_cat}' and '{ground_truth}' are both "
            f"in the '{agent_family}' family.",
        )

    return (
        0.0,
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
        score = (kendall_tau + 1) / 2   →  mapped to [0.0, 1.0]
        1.0 = perfect ranking match
        0.5 = random / uncorrelated ranking
        0.0 = perfectly reversed ranking

    Args:
        action:              Agent's Action object (priority_order must be set).
        ground_truth_order:  Ordered list of defect_ids (highest priority first).

    Returns:
        Tuple of (score, feedback_string).
    """
    agent_order = action.priority_order

    if not agent_order:
        return 0.0, "No priority_order provided for a prioritize action."

    # Build rank dictionaries (0 = highest priority)
    truth_rank = {defect_id: i for i, defect_id in enumerate(ground_truth_order)}

    # Filter agent's list to only IDs present in ground truth to avoid key errors
    common_ids = [did for did in agent_order if did in truth_rank]
    if len(common_ids) < 2:
        return 0.0, "Not enough valid defect IDs in priority_order to compute rank correlation."

    agent_ranks = [agent_order.index(did) for did in common_ids]
    truth_ranks = [truth_rank[did] for did in common_ids]

    tau, p_value = stats.kendalltau(agent_ranks, truth_ranks)

    # Normalise Kendall Tau from [-1, 1] to [0, 1]
    score = round((tau + 1.0) / 2.0, 4)

    if score >= 0.9:
        quality = "Excellent ranking"
    elif score >= 0.7:
        quality = "Good ranking"
    elif score >= 0.5:
        quality = "Moderate ranking"
    else:
        quality = "Poor ranking"

    feedback = (
        f"{quality}: Kendall Tau = {tau:.4f}, normalised score = {score:.4f} "
        f"(p={p_value:.4f}, n={len(common_ids)} items)."
    )
    return score, feedback


# ---------------------------------------------------------------------------
# Grader 3 — Root Cause Diagnosis
# ---------------------------------------------------------------------------

def diagnose_grader(action: Action, ground_truth_cause: str) -> tuple[float, str]:
    """
    Grade a 'diagnose' action.

    Scoring:
        1.0  — exact match with true root cause
        0.4  — same category (machine_issues, human_issues, supply_issues)
        0.0  — wrong

    Args:
        action:              Agent's Action object (root_cause must be set).
        ground_truth_cause:  The true root cause string.

    Returns:
        Tuple of (score, feedback_string).
    """
    agent_cause = action.root_cause

    if agent_cause is None:
        return 0.0, "No root_cause provided for a diagnose action."

    if agent_cause == ground_truth_cause:
        return 1.0, f"Correct diagnosis: '{agent_cause}'."

    agent_category = _ROOT_CAUSE_CATEGORIES.get(agent_cause)
    truth_category = _ROOT_CAUSE_CATEGORIES.get(ground_truth_cause)

    if agent_category is not None and agent_category == truth_category:
        return (
            0.4,
            f"Partially correct: '{agent_cause}' and '{ground_truth_cause}' are both "
            f"in the '{agent_category}' category.",
        )

    return (
        0.0,
        f"Incorrect diagnosis: predicted '{agent_cause}', "
        f"ground truth was '{ground_truth_cause}'.",
    )