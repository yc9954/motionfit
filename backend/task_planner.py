"""Interpret stated task requirements without inventing video semantics or payload mass."""
from __future__ import annotations

import copy
import math
import re
from typing import Any, Dict, Optional

from backend.gemini import ARGON_MODEL, _extract_thought_and_text, _get_genai_client, _parse_json_safely

CONSTRAINT_FIELDS = ("payload_kg", "max_cost_usd", "max_total_mass_kg", "min_reach_m")
SCHEMA = {
    "type": "object", "properties": {
        "summary": {"type": "string"},
        "constraints": {"type": "object", "properties": {key: {"type": ["number", "null"]} for key in CONSTRAINT_FIELDS}, "required": list(CONSTRAINT_FIELDS)},
        "orientation_constraint": {"type": "string", "enum": ["free", "keep_level", "keep_upright"]},
    }, "required": ["summary", "constraints", "orientation_constraint"],
}


def _number(value: Any, key: str) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (key != "payload_kg" and number == 0):
        raise ValueError(f"{key} must be {'nonnegative' if key == 'payload_kg' else 'positive'} and finite")
    if key == "payload_kg" and number > 100:
        raise ValueError("payload_kg exceeds this 4-DOF arm model's 100 kg input limit")
    return number


def _explicit_requirements(prompt: str) -> Dict[str, Optional[float]]:
    """Conservative unit parser. Volume and KRW never silently become mass or USD."""
    out: Dict[str, Optional[float]] = {key: None for key in CONSTRAINT_FIELDS}
    text = prompt.lower().replace(",", "")
    budget = re.search(r"(?:\$|usd\s*)(\d+(?:\.\d+)?)|(?:budget|예산|가격)\s*(?:은|는|:)?\s*(\d+(?:\.\d+)?)\s*(?:달러|usd|\$)", text)
    if budget:
        out["max_cost_usd"] = float(budget.group(1) or budget.group(2))
    reach = re.search(r"(?:reach|도달거리|도달|작업반경|리치)\s*(?:는|은|:)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|미터)", text)
    if reach:
        out["min_reach_m"] = float(reach.group(1)) * {"mm": .001, "cm": .01, "m": 1, "미터": 1}[reach.group(2)]
    masses = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(kg|킬로그램|g|그램)(?![a-z])", text))
    for match in masses:
        prefix = text[max(0, match.start() - 28):match.start()]
        value = round(float(match.group(1)) * (.001 if match.group(2) in ("g", "그램") else 1), 8)
        if re.search(r"(?:total\s*(?:robot\s*)?(?:mass|weight)|전체\s*무게|총\s*(?:무게|질량|중량)|로봇\s*(?:무게|중량))[^,;.\n]*$", prefix):
            out["max_total_mass_kg"] = value
        elif re.search(r"(?:payload|하중|물체|물건|병|짐|중량|질량|무게)[^,;.\n]*$", prefix) or len(masses) == 1:
            out["payload_kg"] = value
    return out


def plan_task(prompt: str, constraints: Optional[Dict[str, Any]] = None, previous: Optional[Dict[str, Any]] = None, orientation: Optional[str] = None) -> Dict[str, Any]:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Describe the task before starting verification")
    requested_axes = re.search(r"\b([1-9])\s*[- ]?(?:dof|joints?|axes)\b|([1-9])\s*축", prompt.lower())
    if requested_axes and int(requested_axes.group(1) or requested_axes.group(2)) != 4:
        raise ValueError("This workspace verifies a 4-DOF arm. Another joint topology needs a corresponding kinematic and dynamic model.")
    old_constraints = (previous or {}).get("constraints", {})
    orientation_hint = "keep_level" if re.search(r"수평|평평|keep.level|horizontal", prompt, re.I) else ("keep_upright" if re.search(r"세워|똑바로|수직|upright", prompt, re.I) else (previous or {}).get("orientation_constraint", "free"))
    parsed = {"summary": prompt, "constraints": _explicit_requirements(prompt), "orientation_constraint": orientation_hint}
    source, failure = "explicit_unit_parser", None
    client = _get_genai_client()
    if client is not None:
        try:
            interaction = client.interactions.create(
                model=ARGON_MODEL,
                input=[{"type": "text", "text": "Extract only requirements explicitly stated in the user's task. Preserve the user's task in summary. Return null for absent numerical requirements. Do not infer payload mass from a video, bottle volume, object type, or density. Do not convert currencies. payload_kg is the manipulated object's mass; max_total_mass_kg is robot mass. Orientation defaults free unless stated. Task:\n" + prompt}],
                generation_config={"thinking_level": "medium"},
                response_format=[{"type": "text", "mime_type": "application/json", "schema": SCHEMA}],
            )
            _, raw = _extract_thought_and_text(interaction)
            live = _parse_json_safely(raw)
            if not live or not isinstance(live.get("constraints"), dict):
                raise ValueError("Model did not return valid task requirements")
            parsed = live
            source = "gemini_structured_task"
        except Exception as exc:
            message = str(exc).lower()
            if "api key not valid" in message or "api_key_invalid" in message:
                failure = "The configured Gemini API key was rejected. Only explicitly written numbers and units were parsed."
            elif "not found" in message or "permission" in message or "not supported" in message:
                failure = "The configured Gemini model is unavailable to this account or request. Only explicitly written numbers and units were parsed."
            else:
                failure = "Task model unavailable; only explicitly written numbers and units were parsed."
    elif source == "explicit_unit_parser":
        failure = "No live task model is enabled. Only explicitly written numbers and units were parsed."
    combined = copy.deepcopy(old_constraints)
    combined.update({key: value for key, value in parsed["constraints"].items() if key in CONSTRAINT_FIELDS and value is not None})
    for key, value in (constraints or {}).items():
        if key not in CONSTRAINT_FIELDS:
            raise ValueError(f"Unknown task constraint: {key}")
        combined[key] = value
    combined = {key: _number(combined.get(key), key) for key in CONSTRAINT_FIELDS}
    chosen_orientation = orientation or parsed.get("orientation_constraint", "free")
    if chosen_orientation not in ("free", "keep_level", "keep_upright"):
        raise ValueError("orientation_constraint must be free, keep_level, or keep_upright")
    return {
        "prompt": prompt, "summary": str(parsed.get("summary") or prompt),
        "constraints": combined, "orientation_constraint": chosen_orientation,
        "interpretation_source": source, "interpretation_notice": failure,
        "missing_fields": ["payload_kg"] if combined["payload_kg"] is None else [],
        "assumptions": [
            "Initial geometry is a configurable 4-DOF yaw/pitch/pitch/pitch arm; it is not inferred from the video.",
            "Unspecified budget, robot mass and minimum reach do not constrain the solver.",
            "Video supplies shoulder-relative arm motion, scaled to the selected arm geometry; payload mass must be supplied by the user.",
            "Verification models an attached tool payload and does not validate grasp, collision, thermal life or mechanical mounting compatibility.",
        ],
    }


def apply_task_to_design(design: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(design)
    requirements = task["constraints"]
    if requirements["payload_kg"] is not None:
        out["payload_kg"] = requirements["payload_kg"]
    for key in ("max_cost_usd", "max_total_mass_kg", "min_reach_m"):
        out["constraints"][key] = requirements[key] if requirements[key] is not None else (0.0 if key == "min_reach_m" else 1e9)
    return out
