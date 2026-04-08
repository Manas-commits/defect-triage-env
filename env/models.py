"""
env/models.py

Pydantic v2 data models for the Manufacturing Defect Triage RL environment.
Defines the three core schemas: Observation, Action, and Reward.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Observation — what the agent SEES each step
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Represents the current state of the manufacturing environment visible
    to the agent at each timestep.
    """

    defect_id: str = Field(
        ...,
        description="Unique identifier for the defect under inspection (e.g. 'DEF-001').",
    )
    defect_description: str = Field(
        ...,
        description="Natural-language description of the defect as recorded at the QC station.",
    )
    sensor_readings: dict[str, float] = Field(
        ...,
        description=(
            "Live sensor readings at the time the defect was captured. "
            "Keys include 'temperature' (°C), 'vibration' (mm/s), 'pressure' (bar)."
        ),
    )
    defect_image_metadata: dict = Field(
        ...,
        description=(
            "Metadata extracted from the defect image. "
            "Contains 'location' (top/bottom/side/inner), "
            "'severity_score' (0.0–1.0), and 'size_mm' (float)."
        ),
    )
    machine_id: str = Field(
        ...,
        description="ID of the machine that produced the defective part (e.g. 'M01').",
    )
    shift: str = Field(
        ...,
        description="Production shift during which the defect occurred: 'morning', 'afternoon', or 'night'.",
    )
    batch_id: str = Field(
        ...,
        description="Batch identifier for the production run (e.g. 'BATCH-2024-001').",
    )
    timestamp: str = Field(
        ...,
        description="ISO-8601 timestamp when the defect was logged.",
    )
    defect_queue: list[dict] = Field(
        default_factory=list,
        description=(
            "List of pending defects awaiting triage. Each entry contains "
            "basic info: defect_id, severity_score, machine_id, and shift."
        ),
    )
    current_task: str = Field(
        ...,
        description=(
            "The task the agent should perform this step. "
            "One of: 'classify', 'prioritize', 'diagnose'."
        ),
    )

    @field_validator("current_task")
    @classmethod
    def validate_current_task(cls, v: str) -> str:
        allowed = {"classify", "prioritize", "diagnose"}
        if v not in allowed:
            raise ValueError(f"current_task must be one of {allowed}, got '{v}'")
        return v

    @field_validator("shift")
    @classmethod
    def validate_shift(cls, v: str) -> str:
        allowed = {"morning", "afternoon", "night"}
        if v not in allowed:
            raise ValueError(f"shift must be one of {allowed}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Action — what the agent CAN DO
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES = {"classify", "prioritize", "diagnose"}
VALID_DEFECT_CATEGORIES = {"dimensional", "surface", "material", "assembly", "cosmetic"}
VALID_ROOT_CAUSES = {
    "tool_wear",
    "calibration_drift",
    "material_defect",
    "operator_error",
    "machine_vibration",
}


class Action(BaseModel):
    """
    Represents a single decision taken by the agent.

    Only one of the optional fields (defect_category, priority_order, root_cause)
    should be populated, depending on the current task.
    """

    action_type: str = Field(
        ...,
        description="The type of action: 'classify', 'prioritize', or 'diagnose'.",
    )
    defect_category: Optional[str] = Field(
        default=None,
        description=(
            "Used when action_type == 'classify'. "
            "One of: 'dimensional', 'surface', 'material', 'assembly', 'cosmetic'."
        ),
    )
    priority_order: Optional[list[str]] = Field(
        default=None,
        description=(
            "Used when action_type == 'prioritize'. "
            "Ordered list of defect_ids from highest to lowest priority."
        ),
    )
    root_cause: Optional[str] = Field(
        default=None,
        description=(
            "Used when action_type == 'diagnose'. "
            "Free-text or one of: 'tool_wear', 'calibration_drift', "
            "'material_defect', 'operator_error', 'machine_vibration'."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent's self-reported confidence in this action (0.0 = uncertain, 1.0 = certain).",
    )

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        if v not in VALID_ACTION_TYPES:
            raise ValueError(f"action_type must be one of {VALID_ACTION_TYPES}, got '{v}'")
        return v

    @field_validator("defect_category")
    @classmethod
    def validate_defect_category(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_DEFECT_CATEGORIES:
            raise ValueError(
                f"defect_category must be one of {VALID_DEFECT_CATEGORIES}, got '{v}'"
            )
        return v


# ---------------------------------------------------------------------------
# Reward — what score the agent receives
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """
    Encapsulates the reward signal returned after each agent step.
    """

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Primary reward signal normalised to [0.0, 1.0].",
    )
    partial_credit: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Partial credit awarded before step penalties / bonuses are applied. "
            "Useful for curriculum learning signals."
        ),
    )
    feedback: str = Field(
        ...,
        description="Human-readable explanation of why this reward was assigned.",
    )
    done: bool = Field(
        ...,
        description="Whether the current episode has ended.",
    )