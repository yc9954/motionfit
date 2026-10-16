"""Gemini 4 Argon & Gemini 3.8 Flash Interactions API client for MotionFit.

Model roles (Section 0):
- Design diagnosis and part selection (core loop): gemini-4-argon, thinking_level="high"
- Motion intent from video: gemini-4-argon, thinking_level="medium"
- Datasheet field extraction: gemini-4-argon, thinking_level="low", structured JSON output
- Report prose generation: gemini-3.8-flash, default
- Schema and unit sanity routing: gemini-3.5-flash-lite, default

Uses google-genai Interactions API (`client.interactions.create`).
Never passes deprecated temperature, top_p, top_k, or thinking_budget.
Uses previous_interaction_id to retain server-side design-loop state.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.catalog_normalize import normalize_catalog
from backend.model_builder import (
    compute_mass_and_cost,
    get_joint_actuator_specs,
    index_catalog,
    load_materials,
)

logger = logging.getLogger("motionfit.gemini")

ARGON_MODEL = os.environ.get("MOTIONFIT_ARGON_MODEL", "gemini-4-argon")
FLASH_MODEL = os.environ.get("MOTIONFIT_FLASH_MODEL", "gemini-3.8-flash")
LITE_MODEL = os.environ.get("MOTIONFIT_LITE_MODEL", "gemini-3.5-flash-lite")
HTTP_TIMEOUT_S = float(os.environ.get("MOTIONFIT_GEMINI_TIMEOUT_S", "45.0"))


# ---------------------------------------------------------------------------
# JSON Schemas for Structured Outputs (Section 5.1, 5.2, 5.4, 5.5)
# ---------------------------------------------------------------------------

PROPOSAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "diagnosis": {
            "type": "string",
            "description": "Concise physics diagnosis of why the design failed or how it can be optimized, citing timestamps, motion phases, and coupled torque effects.",
        },
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Parameter path: J1.actuator_id .. J4.actuator_id, J1.gearbox_id .. J4.gearbox_id, links.upper_arm_m, links.forearm_m, or links.tube_profile.",
                    },
                    "from": {"type": ["string", "number", "null"]},
                    "to": {"type": ["string", "number", "null"]},
                    "rationale": {
                        "type": "string",
                        "description": "Engineering justification with exact datasheet numbers and coupled torque impact.",
                    },
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doc": {"type": "string"},
                                "page": {"type": "integer"},
                                "field": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["doc", "page", "field"],
                        },
                    },
                },
                "required": ["target", "from", "to", "rationale", "citations"],
            },
        },
        "predicted_effect": {
            "type": "object",
            "properties": {
                "j2_peak_ratio": {"type": "number"},
                "j2_rms_ratio": {"type": "number"},
                "total_mass_kg": {"type": "number"},
                "total_cost_usd": {"type": "number"},
            },
            "required": ["j2_peak_ratio", "j2_rms_ratio", "total_mass_kg", "total_cost_usd"],
        },
    },
    "required": ["diagnosis", "changes", "predicted_effect"],
}

INTENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_summary": {"type": "string"},
        "phases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "t_start": {"type": "number"},
                    "t_end": {"type": "number"},
                    "notes": {"type": "string"},
                },
                "required": ["name", "t_start", "t_end", "notes"],
            },
        },
        "orientation_constraint": {
            "type": "string",
            "enum": ["free", "keep_upright", "keep_level"],
        },
        "Inferred_payload_note": {"type": "string"},
    },
    "required": ["task_summary", "phases", "orientation_constraint"],
}

REPORT_PROSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "coupling_analysis": {"type": "string"},
        "recommendation_notes": {"type": "string"},
    },
    "required": ["executive_summary", "coupling_analysis", "recommendation_notes"],
}


# Source-bearing quantities keep units and citations attached to the exact field.
CATALOG_QUANTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": ["number", "null"]}, "unit": {"type": "string"}, "raw": {"type": "string"},
        "src": {"type": "object", "properties": {"doc": {"type": "string"}, "page": {"type": "integer", "minimum": 1}}, "required": ["doc", "page"]},
    },
    "required": ["value", "src"],
}
CATALOG_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {"parts": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "part_id": {"type": "string"}, "type": {"type": "string", "enum": ["actuator", "gearbox"]},
            "vendor": {"type": "string"}, "series": {"type": "string"}, "rated_voltage_v": {"type": "number"},
            "dimensions_mm": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            **{name: CATALOG_QUANTITY_SCHEMA for name in ("stall_torque_nm", "continuous_torque_nm", "no_load_speed_rad_s", "mass_kg", "ratio", "efficiency", "rated_output_torque_nm")},
        },
        "required": ["part_id", "type", "mass_kg"],
    }}},
    "required": ["parts"],
}


DESIGN_LOOP_SYSTEM_INSTRUCTION = """You are Gemini 4 Argon acting as Lead Robotics Hardware Architect for MotionFit.
Your job is to diagnose coupled 4-DOF serial robot arm failures from MuJoCo inverse/forward dynamics and select real replacement actuators, inline planetary gearboxes, or link modifications strictly from the provided catalog.json and materials.json.

STRICT RULES:
1. NEVER invent a part_id, datasheet filename, page number, or specification value. Every part you propose MUST exist in catalog.json with the matching type ("actuator" or "gearbox").
2. Propose between 1 and 3 targeted changes per iteration.
3. Account for coupled physics: upgrading a distal joint (J3 or J4) to a heavier motor increases the gravity and inertial moment on the shoulder pitch joint (J2).
4. Every hardware change must include at least one citation with {"doc": "<pdf filename>", "page": <integer>, "field": "<spec field>", "quote": "<exact spec quote>"}.
5. Ensure the resulting design stays within constraints.max_total_mass_kg, constraints.max_cost_usd, and constraints.min_reach_m.
6. Output strictly valid JSON matching the requested schema."""


def _get_genai_client() -> Optional[Any]:
    """Instantiate google.genai.Client if API key is available and offline mode is not forced."""
    if os.environ.get("MOTIONFIT_OFFLINE", "0") == "1":
        return None
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai

        return genai.Client(api_key=api_key, http_options={"timeout": int(HTTP_TIMEOUT_S * 1000)})
    except Exception as exc:
        logger.warning("Could not initialize google.genai.Client: %s", exc)
        return None


def _extract_thought_and_text(interaction: Any) -> Tuple[str, str]:
    """Read the installed SDK's model_output and public thought-summary steps."""
    def get(obj: Any, key: str, default: Any = None) -> Any:
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    thought_parts: List[str] = []
    output_parts: List[str] = []
    for step in get(interaction, "steps", []) or []:
        if get(step, "type") == "thought":
            summary = get(step, "summary", []) or []
            if isinstance(summary, str):
                thought_parts.append(summary)
            else:
                thought_parts.extend(str(get(c, "text")) for c in summary if get(c, "type") == "text" and get(c, "text"))
        elif get(step, "type") == "model_output":
            output_parts.extend(str(get(c, "text")) for c in get(step, "content", []) or [] if get(c, "type") == "text" and get(c, "text"))
    output_text = get(interaction, "output_text") or "\n".join(output_parts)
    return "\n".join(thought_parts).strip(), str(output_text).strip()


def _parse_json_safely(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from model output text, stripping markdown code fences if present."""
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                return None
    return None


# ---------------------------------------------------------------------------
# Physics-Guided Catalog Search (Measured-Load Screening)
# ---------------------------------------------------------------------------

def _physics_guided_proposal(
    design: Dict[str, Any],
    check_result: Dict[str, Any],
    metrics: Dict[str, Any],
    catalog: Any,
    intent: Optional[Dict[str, Any]] = None,
    optimize_mode: bool = False,
) -> Dict[str, Any]:
    """Rank catalog combinations using measured loads; MuJoCo adjudicates each proposal.

    The estimates intentionally use peak speed and peak torque together. They are
    conservative screening estimates, not a substitute for re-simulation or a
    claim that a gearbox physically mates to an actuator.
    """
    import copy
    import math

    from backend.model_builder import available_torque

    cat_map = index_catalog(catalog)
    materials = load_materials()
    constraints = design.get("constraints", {})
    thresholds = design.get("thresholds", {})
    peak_limit = float(thresholds.get("peak_torque_ratio", 0.8))
    rms_limit = float(thresholds.get("rms_torque_ratio", 1.0))
    speed_limit = float(thresholds.get("peak_speed_ratio", 0.9))
    max_mass = float(constraints.get("max_total_mass_kg", float("inf")))
    max_cost = float(constraints.get("max_cost_usd", float("inf")))
    working = copy.deepcopy(design)
    observed = metrics.get("joints", {})
    failing = [j for j in ("J1", "J2", "J3", "J4") if (
        float(observed.get(j, {}).get("peak_ratio", 0)) > peak_limit
        or float(observed.get(j, {}).get("rms_ratio", 0)) > rms_limit
        or float(observed.get(j, {}).get("speed_ratio", 0)) > speed_limit)]
    # Highest normalized overload first, independently of a particular robot.
    failing.sort(key=lambda j: max(float(observed[j].get("peak_ratio", 0)) / peak_limit,
                                  float(observed[j].get("rms_ratio", 0)) / rms_limit,
                                  float(observed[j].get("speed_ratio", 0)) / speed_limit), reverse=True)
    joint_order = list(observed) if optimize_mode and check_result.get("passed") else failing
    changes: List[Dict[str, Any]] = []
    messages: List[str] = []
    estimates = {j: dict(observed[j]) for j in observed}

    def citation(part: Dict[str, Any], field: str) -> Dict[str, Any]:
        value = part.get(field) or {}
        src = value.get("src") or {}
        return {"doc": src.get("doc"), "page": src.get("page"), "field": field,
                "quote": str(value.get("raw") or f"{field}: {value.get('value')}")}

    actuators = [p for p in cat_map.values() if p.get("type") == "actuator"]
    gearboxes = [None] + [p for p in cat_map.values() if p.get("type") == "gearbox"]
    for joint in joint_order:
        if len(changes) >= 3:
            break
        jm = observed[joint]
        if not all(key in jm for key in ("peak_req_nm", "rms_req_nm", "peak_speed_rad_s")):
            continue
        current = working["joints"][joint]
        bus_voltage = cat_map.get(current["actuator_id"], {}).get("rated_voltage_v")
        before = compute_mass_and_cost(working, cat_map, materials)
        candidates = []
        for actuator in actuators:
            if (bus_voltage is not None and actuator.get("rated_voltage_v") is not None
                    and abs(float(actuator["rated_voltage_v"]) - float(bus_voltage)) > 1e-6):
                continue
            for gearbox in gearboxes:
                cfg = {"actuator_id": actuator["part_id"], "gearbox_id": gearbox["part_id"] if gearbox else None}
                fields = [field for field in ("actuator_id", "gearbox_id") if cfg[field] != current.get(field)]
                if not fields or len(changes) + len(fields) > 3:
                    continue
                try:
                    spec = get_joint_actuator_specs(cfg, cat_map)
                    if min(spec["tau_stall_eff"], spec["tau_cont_eff"], spec["omega_nl_eff"]) <= 0:
                        continue
                    avail = available_torque(float(jm["peak_speed_rad_s"]), spec["tau_stall_eff"], spec["tau_rated_eff"], spec["omega_nl_eff"])
                    peak = float(jm["peak_req_nm"]) / max(avail, 1e-9)
                    rms = float(jm["rms_req_nm"]) / spec["tau_cont_eff"]
                    speed = float(jm["peak_speed_rad_s"]) / spec["omega_nl_eff"]
                    if peak > peak_limit * 0.98 or rms > rms_limit * 0.98 or speed > speed_limit * 0.98:
                        continue
                    candidate = copy.deepcopy(working)
                    candidate["joints"][joint] = cfg
                    summary = compute_mass_and_cost(candidate, cat_map, materials)
                    if summary["total_mass_kg"] > max_mass or summary["total_cost_usd"] > max_cost:
                        continue
                    score = summary["total_cost_usd"] + 200.0 * summary["total_mass_kg"]
                    old_score = before["total_cost_usd"] + 200.0 * before["total_mass_kg"]
                    if optimize_mode and score >= old_score - 1e-6:
                        continue
                    if not all(math.isfinite(x) for x in (score, peak, rms, speed)):
                        continue
                    candidates.append((score, cfg, fields, peak, rms, speed, summary))
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    continue
        if not candidates:
            messages.append(f"{joint}: no catalog combination satisfies the measured torque/speed demand and current budgets.")
            continue
        _, cfg, fields, peak, rms, speed, _ = min(candidates, key=lambda row: row[0])
        for field in fields:
            part = cat_map.get(cfg[field])
            spec_field = "stall_torque_nm" if field == "actuator_id" else "ratio"
            changes.append({
                "target": f"{joint}.{field}", "from": current.get(field), "to": cfg[field],
                "rationale": (f"Catalog screening for {joint} uses measured peak torque {jm['peak_req_nm']:.4f} N·m, "
                              f"RMS torque {jm['rms_req_nm']:.4f} N·m and peak speed {jm['peak_speed_rad_s']:.4f} rad/s. "
                              "This candidate requires a new coupled simulation; mechanical mounting compatibility remains unverified."),
                "citations": [citation(part, spec_field)] if part else [],
            })
        working["joints"][joint] = cfg
        estimates[joint].update(peak_ratio=peak, rms_ratio=rms, speed_ratio=speed)
        messages.append(f"{joint}: screened {len(candidates)} budget-feasible combinations against the measured load; selected {cfg['actuator_id']}" + (f" with {cfg['gearbox_id']}." if cfg['gearbox_id'] else "."))
        if optimize_mode:
            break
    summary = compute_mass_and_cost(working, cat_map, materials)
    if not messages:
        messages = ["No hardware change is justified by the measured joint loads. Remaining violations require task, reach, or constraint changes."]
    return {
        "diagnosis": " ".join(messages),
        "changes": changes,
        "predicted_effect": {
            "j2_peak_ratio": round(float(estimates.get("J2", {}).get("peak_ratio", 0)), 4),
            "j2_rms_ratio": round(float(estimates.get("J2", {}).get("rms_ratio", 0)), 4),
            "total_mass_kg": summary["total_mass_kg"],
            "total_cost_usd": summary["total_cost_usd"],
        },
    }


# ---------------------------------------------------------------------------
# Public API Functions (Section 0, 5.1, 5.2, 5.4, 5.5)
# ---------------------------------------------------------------------------

def propose_design_changes(
    design: Dict[str, Any],
    check_result: Dict[str, Any],
    metrics: Dict[str, Any],
    catalog: Any,
    intent: Optional[Dict[str, Any]] = None,
    previous_interaction_id: Optional[str] = None,
    is_first_turn: bool = True,
    optimize_mode: bool = False,
) -> Dict[str, Any]:
    """Call Gemini 4 Argon (thinking_level='high') via Interactions API to diagnose and propose design changes.

    Returns:
      {
        "proposal": Dict[str, Any],
        "interaction_id": str,
        "thought_summary": str,
        "model": str,
        "used_live_api": bool,
      }
    """
    client = _get_genai_client()
    materials = load_materials()

    mode_prompt = (
        "POST-PASS OPTIMIZATION MODE: All 17 checks currently pass. Propose 1 to 2 changes from catalog.json "
        "(such as pairing a lighter motor with an inline planetary gearbox) that reduce total_mass_kg or total_cost_usd "
        "while keeping all torque, speed, and RMSE checks strictly passing."
        if optimize_mode
        else "FAILURE RECOVERY MODE: Diagnose the failing joints and propose 1 to 3 changes from catalog.json "
        "that resolve all violations without exceeding mass or cost budgets."
    )

    if is_first_turn or not previous_interaction_id:
        prompt_payload = {
            "mode": mode_prompt,
            "current_design": design,
            "checker_result": check_result,
            "simulation_metrics": metrics,
            "motion_intent": intent,
            "catalog": catalog,
            "materials": materials,
        }
    else:
        prompt_payload = {
            "mode": mode_prompt,
            "updated_design": design,
            "checker_result": check_result,
            "simulation_metrics": metrics,
        }

    if client is not None:
        try:
            kwargs: Dict[str, Any] = {
                "model": ARGON_MODEL,
                "system_instruction": DESIGN_LOOP_SYSTEM_INSTRUCTION,
                "input": json.dumps(prompt_payload, indent=2),
                "generation_config": {
                    "thinking_level": "high",
                    "thinking_summaries": "auto",
                },
                "response_format": [{"type": "text", "mime_type": "application/json", "schema": PROPOSAL_SCHEMA}],
            }
            if previous_interaction_id:
                kwargs["previous_interaction_id"] = previous_interaction_id

            interaction = client.interactions.create(**kwargs)
            thought_summary, out_text = _extract_thought_and_text(interaction)
            parsed = _parse_json_safely(out_text)
            if parsed and "changes" in parsed and "diagnosis" in parsed:
                return {
                    "proposal": parsed,
                    "interaction_id": getattr(interaction, "id", None) or f"int_{uuid.uuid4().hex[:10]}",
                    "thought_summary": thought_summary or parsed.get("diagnosis", ""),
                    "model": ARGON_MODEL,
                    "used_live_api": True,
                }
        except Exception as exc:
            logger.warning("Gemini 4 Argon Interactions API call fell back to physics solver: %s", exc)

    fallback_proposal = _physics_guided_proposal(
        design=design,
        check_result=check_result,
        metrics=metrics,
        catalog=catalog,
        intent=intent,
        optimize_mode=optimize_mode,
    )
    return {
        "proposal": fallback_proposal,
        "interaction_id": previous_interaction_id,
        "thought_summary": fallback_proposal["diagnosis"],
        "model": "deterministic_catalog_search",
        "used_live_api": False,
        "source": "measured_load_catalog_screening",
        "fallback_reason": "Live model unavailable or returned an invalid response.",
    }


def repair_design_proposal(
    validation_errors: List[str],
    previous_interaction_id: Optional[str],
    design: Dict[str, Any],
    check_result: Dict[str, Any],
    metrics: Dict[str, Any],
    catalog: Any,
    intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send validation errors back to Gemini 4 Argon on the same previous_interaction_id for repair."""
    client = _get_genai_client()
    if client is not None and previous_interaction_id:
        try:
            repair_input = (
                "Your previous proposal failed deterministic validation in validate.py with the following errors:\n"
                + "\n".join(f"- {e}" for e in validation_errors)
                + "\n\nPlease output a corrected JSON proposal using ONLY valid part_ids and exact citations from catalog.json."
            )
            interaction = client.interactions.create(
                model=ARGON_MODEL,
                system_instruction=DESIGN_LOOP_SYSTEM_INSTRUCTION,
                input=repair_input,
                previous_interaction_id=previous_interaction_id,
                generation_config={
                    "thinking_level": "high",
                    "thinking_summaries": "auto",
                },
                response_format=[{"type": "text", "mime_type": "application/json", "schema": PROPOSAL_SCHEMA}],
            )
            thought_summary, out_text = _extract_thought_and_text(interaction)
            parsed = _parse_json_safely(out_text)
            if parsed and "changes" in parsed:
                return {
                    "proposal": parsed,
                    "interaction_id": getattr(interaction, "id", None) or previous_interaction_id,
                    "thought_summary": thought_summary or "Repaired proposal against catalog.json constraints.",
                    "model": ARGON_MODEL,
                    "used_live_api": True,
                }
        except Exception as exc:
            logger.warning("Gemini 4 Argon repair call fell back to physics solver: %s", exc)

    fallback_proposal = _physics_guided_proposal(
        design=design,
        check_result=check_result,
        metrics=metrics,
        catalog=catalog,
        intent=intent,
        optimize_mode=False,
    )
    return {
        "proposal": fallback_proposal,
        "interaction_id": previous_interaction_id,
        "thought_summary": fallback_proposal["diagnosis"],
        "model": "deterministic_catalog_search",
        "used_live_api": False,
        "source": "measured_load_catalog_screening",
    }


def analyze_motion_intent(
    landmarks_summary: Dict[str, Any],
    video_bytes: Optional[bytes] = None,
    video_mime_type: str = "video/mp4",
    cached_intent_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Analyze motion video + landmark trajectory with Gemini 4 Argon (thinking_level='medium')."""
    if cached_intent_path and cached_intent_path.exists():
        try:
            return json.loads(cached_intent_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    client = _get_genai_client()
    if client is not None:
        try:
            inputs: List[Dict[str, Any]] = []
            if video_bytes:
                inputs.append(
                    {
                        "type": "video",
                        "mime_type": video_mime_type,
                        "data": base64.b64encode(video_bytes).decode("ascii"),
                    }
                )
            inputs.append(
                {
                    "type": "text",
                    "text": (
                        "Segment this single-arm manipulation motion into distinct physical phases with start/end "
                        "timestamps, identify peak reach or speed moments, and determine the end-effector orientation "
                        f"constraint. Trajectory summary:\n{json.dumps(landmarks_summary, indent=2)}"
                    ),
                }
            )
            interaction = client.interactions.create(
                model=ARGON_MODEL,
                input=inputs,
                generation_config={"thinking_level": "medium"},
                response_format=[{"type": "text", "mime_type": "application/json", "schema": INTENT_SCHEMA}],
            )
            _, out_text = _extract_thought_and_text(interaction)
            parsed = _parse_json_safely(out_text)
            if parsed and "phases" in parsed:
                parsed["source"] = "gemini_video" if video_bytes else "gemini_landmark_summary"
                parsed["used_live_api"] = True
                parsed["model"] = ARGON_MODEL
                return parsed
        except Exception as exc:
            logger.warning("Gemini 4 Argon motion intent fell back to kinematic phase segmentation: %s", exc)

    duration_s = float(landmarks_summary.get("duration_s", 0.0))
    return {
        "task_summary": f"Recorded single-arm trajectory ({duration_s:.3f} s). Semantic task phases require video analysis.",
        "phases": [{"name": "recorded_motion", "t_start": 0.0, "t_end": duration_s,
                    "notes": "Measured kinematics only; no object, payload, or grasp inference."}],
        "orientation_constraint": "free",
        "Inferred_payload_note": "Payload must be supplied by the user; it cannot be inferred from pose coordinates.",
        "source": "measured_kinematics_only",
        "used_live_api": False,
    }


def extract_catalog_from_pdfs(
    pdf_paths: List[Path],
    prices_path: Optional[Path] = None,
    cached_catalog_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Extract actuator/gearbox catalog from datasheet PDFs using Gemini 4 Argon (thinking_level='low')
    followed by deterministic normalization in catalog_normalize.py.
    """
    if cached_catalog_path and cached_catalog_path.exists():
        raw_cat = json.loads(cached_catalog_path.read_text(encoding="utf-8"))
        return normalize_catalog(raw_cat, prices_path=prices_path)

    client = _get_genai_client()
    extracted_parts: List[Dict[str, Any]] = []
    errors: List[str] = []
    if not pdf_paths:
        raise ValueError("No datasheet PDFs were supplied.")
    if client is None:
        raise ValueError("Datasheet extraction requires an available Gemini API client; no catalog was substituted.")

    if client is not None and pdf_paths:
        for pdf_path in pdf_paths:
            try:
                from pypdf import PdfReader
                page_count = len(PdfReader(str(pdf_path)).pages)
                pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
                interaction = client.interactions.create(
                    model=ARGON_MODEL,
                    input=[
                        {
                            "type": "document",
                            "mime_type": "application/pdf",
                            "data": pdf_b64,
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Extract all actuator or gearbox specifications from datasheet '{pdf_path.name}'. "
                                "Include exact page numbers and the exact raw quotation for every numeric field. "
                                "The value must agree with its explicit unit; use SI units for numeric values when unit is omitted. "
                                "Use null for an unstated continuous thermal rating, never infer it from stall torque. "
                                "Do not infer mounting compatibility from housing dimensions. Prefer 12.0V specifications if multiple voltages are listed."
                            ),
                        },
                    ],
                    generation_config={"thinking_level": "low"},
                    response_format=[{"type": "text", "mime_type": "application/json", "schema": CATALOG_EXTRACTION_SCHEMA}],
                )
                _, out_text = _extract_thought_and_text(interaction)
                parsed = _parse_json_safely(out_text)
                parts = parsed.get("parts", []) if isinstance(parsed, dict) else []
                if not isinstance(parts, list) or not parts:
                    raise ValueError("No valid catalog parts were returned")
                for part in parts:
                    if not isinstance(part, dict) or not part.get("part_id") or part.get("type") not in ("actuator", "gearbox"):
                        raise ValueError("Each extracted part requires a part_id and actuator/gearbox type")
                    required = ("stall_torque_nm", "no_load_speed_rad_s", "mass_kg") if part["type"] == "actuator" else ("ratio", "efficiency", "rated_output_torque_nm", "mass_kg")
                    for field in required:
                        quantity = part.get(field)
                        if not isinstance(quantity, dict) or quantity.get("value") is None:
                            raise ValueError(f"{part['part_id']}: missing {field}")
                    for field, quantity in part.items():
                        if isinstance(quantity, dict) and "value" in quantity and field != "price_usd":
                            src = quantity.get("src") or {}
                            page = src.get("page")
                            if src.get("doc") != pdf_path.name or not isinstance(page, int) or not (1 <= page <= page_count):
                                raise ValueError(f"{part['part_id']}.{field}: source must match the supplied document and an existing page")
                    part["provenance"] = {"source": "user_pdf_gemini_extraction", "requires_spec_review": True, "document": pdf_path.name}
                extracted_parts.extend(parts)
            except Exception as exc:
                errors.append(f"{pdf_path.name}: {exc}")
                logger.warning("PDF extraction failed for %s: %s", pdf_path.name, exc)

    if errors or not extracted_parts:
        raise ValueError("Datasheet extraction failed: " + "; ".join(errors or ["No catalog parts were extracted"]))
    ids = [part["part_id"] for part in extracted_parts]
    if len(set(ids)) != len(ids):
        raise ValueError("Datasheet extraction returned duplicate part IDs; resolve document/voltage variants before confirming.")
    return normalize_catalog({
        "parts": extracted_parts,
        "provenance": {"source": "user_pdf_gemini_extraction", "used_live_api": True, "model": ARGON_MODEL,
                       "documents": [path.name for path in pdf_paths], "requires_spec_review": True},
    }, prices_path=prices_path)


def generate_report_prose(
    initial_design: Dict[str, Any],
    final_design: Dict[str, Any],
    iterations: List[Dict[str, Any]],
    intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Generate concise engineering report prose using Gemini 3.8 Flash (`gemini-3.8-flash`)."""
    return {
        "executive_summary": "The report compares the submitted design with the selected simulation result. Numerical results are taken directly from the verification records.",
        "coupling_analysis": "In a serial arm, distal actuator and payload masses contribute to upstream gravity and inertia. Each candidate must be re-simulated after a hardware change.",
        "recommendation_notes": "The rigid arm simulation uses a fixed end-effector payload. Grasp contact, collision, mechanical mounting, power electronics, stiffness, backlash and thermal dissipation remain outside this verification scope.",
    }
