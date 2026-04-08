"""
env/tasks.py

Synthetic data generator for the Manufacturing Defect Triage RL environment.
Produces realistic fake data for a metal-parts factory with hidden ground truth.

All random generation uses seed=42 for reproducibility.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MACHINES = ["M01", "M02", "M03", "M04", "M05"]
SHIFTS = ["morning", "afternoon", "night"]
CATEGORIES = ["dimensional", "surface", "material", "assembly", "cosmetic"]
ROOT_CAUSES = [
    "tool_wear",
    "calibration_drift",
    "material_defect",
    "operator_error",
    "machine_vibration",
]
LOCATIONS = ["top", "bottom", "side", "inner"]

# Realistic defect description templates keyed by true category
_DESCRIPTION_TEMPLATES: dict[str, list[str]] = {
    "dimensional": [
        "Bore diameter {delta:.2f}mm out of tolerance at station {station}",
        "Wall thickness variation of {delta:.2f}mm detected on bearing housing",
        "Shaft length exceeds spec by {delta:.2f}mm at QC station {station}",
        "Out-of-roundness {delta:.2f}mm on inner race, detected post-grind",
    ],
    "surface": [
        "Surface scratch {delta:.1f}mm on bearing housing, found at QC station {station}",
        "Pitting corrosion cluster on flange face, depth ~{delta:.2f}mm",
        "Grinding burn marks covering {delta:.0f}% of journal surface",
        "Micro-cracks ({delta:.2f}mm spacing) on hardened outer race",
    ],
    "material": [
        "Porosity detected in casting: {delta:.0f} voids > 0.3mm in cross-section",
        "Hardness below spec ({delta:.1f} HRC) on case-hardened surface",
        "Inclusion cluster ({delta:.2f}mm) found in ultrasonic scan at station {station}",
        "Delamination in sintered insert, area ~{delta:.1f}mm²",
    ],
    "assembly": [
        "Press-fit interference insufficient: {delta:.3f}mm gap at mating face",
        "Fastener torque {delta:.1f} Nm below minimum spec at station {station}",
        "Misalignment of sub-assembly: {delta:.2f}° angular offset",
        "Missing retaining clip detected on shaft assembly at station {station}",
    ],
    "cosmetic": [
        "Paint blister {delta:.1f}mm diameter on exterior housing",
        "Burr {delta:.2f}mm on chamfer edge, cosmetic only",
        "Laser-etch marking faint/incomplete at station {station}",
        "Anodising colour inconsistency across {delta:.0f}% of surface",
    ],
}

# Root-cause hints embedded in sensor readings for task 3
_ROOT_CAUSE_SENSOR_BIAS: dict[str, dict[str, tuple[float, float]]] = {
    "tool_wear":        {"vibration": (3.0, 5.0), "temperature": (220, 250), "pressure": (95, 105)},
    "calibration_drift":{"vibration": (0.1, 1.0), "temperature": (180, 200), "pressure": (105, 110)},
    "material_defect":  {"vibration": (0.5, 2.0), "temperature": (195, 215), "pressure": (90, 100)},
    "operator_error":   {"vibration": (1.0, 3.0), "temperature": (185, 225), "pressure": (90, 110)},
    "machine_vibration":{"vibration": (3.5, 5.0), "temperature": (200, 240), "pressure": (92, 108)},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> random.Random:
    """Return a seeded Random instance."""
    return random.Random(seed)


def _iso_timestamp(base: datetime, offset_minutes: int) -> str:
    return (base + timedelta(minutes=offset_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _generate_defect(
    rng: random.Random,
    defect_index: int,
    batch_id: str,
    base_time: datetime,
    forced_root_cause: str | None = None,
) -> dict[str, Any]:
    """Generate a single realistic defect dict including hidden ground truth."""

    true_category: str = rng.choice(CATEGORIES)
    true_root_cause: str = forced_root_cause if forced_root_cause else rng.choice(ROOT_CAUSES)
    machine_id: str = rng.choice(MACHINES)
    shift: str = rng.choice(SHIFTS)

    # Sensor readings — biased by root cause
    bias = _ROOT_CAUSE_SENSOR_BIAS[true_root_cause]
    temperature = round(rng.uniform(*bias["temperature"]), 2)
    vibration = round(rng.uniform(*bias["vibration"]), 3)
    pressure = round(rng.uniform(*bias["pressure"]), 2)

    # Image metadata
    severity_score = round(rng.uniform(0.0, 1.0), 3)
    size_mm = round(rng.uniform(0.1, 50.0), 2)
    location = rng.choice(LOCATIONS)

    # Priority score — weighted by severity and size
    true_priority_score = round(
        0.6 * severity_score + 0.4 * min(size_mm / 50.0, 1.0), 3
    )

    # Description
    templates = _DESCRIPTION_TEMPLATES[true_category]
    template = rng.choice(templates)
    delta = rng.uniform(0.05, size_mm)
    station = rng.randint(1, 6)
    description = template.format(delta=delta, station=station)

    defect_id = f"DEF-{defect_index:03d}"

    return {
        "defect_id": defect_id,
        "defect_description": description,
        "sensor_readings": {
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
        },
        "defect_image_metadata": {
            "location": location,
            "severity_score": severity_score,
            "size_mm": size_mm,
        },
        "machine_id": machine_id,
        "shift": shift,
        "batch_id": batch_id,
        "timestamp": _iso_timestamp(base_time, defect_index * 15),
        # --- Hidden ground truth (NOT exposed to agent via Observation) ---
        "true_category": true_category,
        "true_priority_score": true_priority_score,
        "true_root_cause": true_root_cause,
    }


def _public_fields(defect: dict[str, Any]) -> dict[str, Any]:
    """Strip hidden ground-truth fields for use in the defect_queue."""
    hidden = {"true_category", "true_priority_score", "true_root_cause"}
    return {k: v for k, v in defect.items() if k not in hidden}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_defect_batch(batch_size: int = 20, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate a batch of synthetic manufacturing defects.

    Args:
        batch_size: Number of defects to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of defect dicts, each containing both public fields and hidden
        ground-truth fields (true_category, true_priority_score, true_root_cause).
    """
    rng = _rng(seed)
    base_time = datetime(2024, 6, 1, 6, 0, 0)
    batch_id = f"BATCH-2024-{rng.randint(1, 999):03d}"
    return [
        _generate_defect(rng, i + 1, batch_id, base_time)
        for i in range(batch_size)
    ]


def get_task_1_scenario(seed: int = 42) -> dict[str, Any]:
    """
    Task 1 — Defect Classification.

    Returns a single defect observation for the agent to classify.
    The hidden field 'true_category' is the ground truth label.
    """
    rng = _rng(seed)
    base_time = datetime(2024, 6, 1, 8, 0, 0)
    batch_id = "BATCH-2024-T1"
    defect = _generate_defect(rng, 1, batch_id, base_time)
    return defect


def get_task_2_scenario(seed: int = 42) -> list[dict[str, Any]]:
    """
    Task 2 — Defect Queue Prioritization.

    Returns a queue of 10 defects. The agent must output an optimal repair
    priority order. Ground truth ranking is derived from 'true_priority_score'
    (descending).
    """
    rng = _rng(seed + 100)  # separate seed offset to keep tasks independent
    base_time = datetime(2024, 6, 1, 14, 0, 0)
    batch_id = "BATCH-2024-T2"
    defects = [
        _generate_defect(rng, i + 1, batch_id, base_time)
        for i in range(10)
    ]
    return defects


def get_task_3_scenario(seed: int = 42) -> dict[str, Any]:
    """
    Task 3 — Root Cause Diagnosis.

    Returns a pattern of 15 defects across 3 shifts and 5 machines that all
    share one dominant root cause. The agent must identify that root cause.

    Returns a dict with keys:
        - 'defects': list of 15 defect dicts
        - 'true_root_cause': the single root cause hidden in the pattern
    """
    rng = _rng(seed + 200)  # separate seed offset
    base_time = datetime(2024, 6, 1, 6, 0, 0)
    batch_id = "BATCH-2024-T3"
    true_root_cause = rng.choice(ROOT_CAUSES)

    defects = []
    for i in range(15):
        defect = _generate_defect(
            rng, i + 1, batch_id, base_time, forced_root_cause=true_root_cause
        )
        defects.append(defect)

    return {
        "defects": defects,
        "true_root_cause": true_root_cause,
    }


def get_ground_truth_priority_order(defects: list[dict[str, Any]]) -> list[str]:
    """
    Compute the ground-truth priority order for a list of defects.

    Sorts by true_priority_score descending, returns ordered list of defect_ids.
    """
    sorted_defects = sorted(defects, key=lambda d: d["true_priority_score"], reverse=True)
    return [d["defect_id"] for d in sorted_defects]