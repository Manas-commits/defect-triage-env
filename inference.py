"""
inference.py  (ROOT level)

Baseline inference script for Manufacturing Defect Triage.
Follows the mandatory [START] / [STEP] / [END] stdout format required by the competition.

Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
"""

import json
import os
import textwrap
from typing import Optional

from openai import OpenAI

from env.environment import ManufacturingDefectEnv
from env.models import Action

# ---------------------------------------------------------------------------
# Credentials & config
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "manufacturing-defect-triage"
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Mandatory stdout log helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert manufacturing quality engineer triaging defects on a metal-parts production line.
    You will receive defect observations and must respond with a valid JSON action object.
    Respond ONLY with raw JSON — no markdown, no explanation, no code fences.

    Action schema:
    {
      "action_type": "classify" | "prioritize" | "diagnose",
      "defect_category": "dimensional" | "surface" | "material" | "assembly" | "cosmetic" | null,
      "priority_order": ["DEF-001", "DEF-002", ...] | null,
      "root_cause": "tool_wear" | "calibration_drift" | "material_defect" | "operator_error" | "machine_vibration" | null,
      "confidence": 0.0 to 1.0
    }
""").strip()


# ---------------------------------------------------------------------------
# Prompt builders per task
# ---------------------------------------------------------------------------

def _build_prompt_task1(obs: dict) -> str:
    return textwrap.dedent(f"""
        TASK: Classify the defect type.
        Defect ID: {obs['defect_id']}
        Description: {obs['defect_description']}
        Machine: {obs['machine_id']}  Shift: {obs['shift']}
        Sensor readings: temperature={obs['sensor_readings']['temperature']}°C, \
vibration={obs['sensor_readings']['vibration']}mm/s, \
pressure={obs['sensor_readings']['pressure']}bar
        Image metadata: location={obs['defect_image_metadata']['location']}, \
severity={obs['defect_image_metadata']['severity_score']:.2f}, \
size={obs['defect_image_metadata']['size_mm']:.1f}mm

        Set action_type = "classify" and defect_category to one of:
        dimensional, surface, material, assembly, cosmetic.
    """).strip()


def _build_prompt_task2(obs: dict) -> str:
    queue = obs.get("defect_queue", [])
    queue_str = "\n".join(
        f"  {d['defect_id']}  machine={d['machine_id']}  shift={d['shift']}  "
        f"severity={d['severity_score']:.2f}"
        for d in queue
    )
    return textwrap.dedent(f"""
        TASK: Prioritize the repair queue from highest to lowest priority.
        Queue ({len(queue)} defects):
        {queue_str}

        Set action_type = "prioritize" and priority_order to an ordered list of all defect IDs.
    """).strip()


def _build_prompt_task3(obs: dict) -> str:
    queue = obs.get("defect_queue", [])
    pattern_str = "\n".join(
        f"  {d['defect_id']}  machine={d['machine_id']}  shift={d['shift']}  "
        f"temp={d['sensor_readings']['temperature']}°C  "
        f"vib={d['sensor_readings']['vibration']}mm/s"
        for d in queue
    )
    return textwrap.dedent(f"""
        TASK: Diagnose the root cause from the defect pattern below.
        Pattern ({len(queue)} defects across shifts and machines):
        {pattern_str}

        Set action_type = "diagnose" and root_cause to one of:
        tool_wear, calibration_drift, material_defect, operator_error, machine_vibration.
    """).strip()


_PROMPT_BUILDERS = {
    "task_1_classify":   _build_prompt_task1,
    "task_2_prioritize": _build_prompt_task2,
    "task_3_diagnose":   _build_prompt_task3,
}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_model(client: OpenAI, user_prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return ""


def _parse_action(raw: str) -> Optional[Action]:
    cleaned = raw.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(l for l in lines if not l.startswith("```"))
    try:
        data = json.loads(cleaned.strip())
        return Action(**data)
    except Exception as exc:
        print(f"[DEBUG] Action parse error: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str, seed: int = 42) -> float:
    env = ManufacturingDefectEnv(task_id=task_id, seed=seed)
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        obs_dict = obs.model_dump()

        prompt = _PROMPT_BUILDERS[task_id](obs_dict)
        raw = _call_model(client, prompt)
        action = _parse_action(raw)

        step = 1
        error_msg = None

        if action is None:
            error_msg = "Failed to parse model response into Action"
            reward_val = 0.0
            done = True
            log_step(step=step, action=str(raw)[:80], reward=reward_val, done=done, error=error_msg)
            rewards.append(reward_val)
            steps_taken = step
        else:
            try:
                _, reward_obj, done, info = env.step(action)
                reward_val = reward_obj.score
                error_msg = info.get("hack_detected", None)
            except Exception as exc:
                reward_val = 0.0
                done = True
                error_msg = str(exc)

            action_str = f"{action.action_type}({action.defect_category or action.root_cause or 'ordered'})"
            log_step(step=step, action=action_str, reward=reward_val, done=done, error=error_msg)
            rewards.append(reward_val)
            steps_taken = step

        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    task_ids = ["task_1_classify", "task_2_prioritize", "task_3_diagnose"]
    scores: dict[str, float] = {}

    for task_id in task_ids:
        scores[task_id] = run_task(client, task_id, seed=42)

    avg = sum(scores.values()) / len(scores)

    print("", flush=True)
    print("=" * 50, flush=True)
    print("  SCORE REPORT", flush=True)
    print("=" * 50, flush=True)
    print(f"  Task 1 (Classify):    {scores.get('task_1_classify', 0):.4f}", flush=True)
    print(f"  Task 2 (Prioritize):  {scores.get('task_2_prioritize', 0):.4f}", flush=True)
    print(f"  Task 3 (Diagnose):    {scores.get('task_3_diagnose', 0):.4f}", flush=True)
    print(f"  Average:              {avg:.4f}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()