"""Deterministic proposal validator and design mutator (Section 5.4)."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.model_builder import compute_mass_and_cost, index_catalog, load_materials


VALID_TARGETS = {
    "J1.actuator_id",
    "J2.actuator_id",
    "J3.actuator_id",
    "J4.actuator_id",
    "J1.gearbox_id",
    "J2.gearbox_id",
    "J3.gearbox_id",
    "J4.gearbox_id",
    "links.upper_arm_m",
    "links.forearm_m",
    "links.tube_profile",
}


def apply_changes_to_design(
    design: Dict[str, Any],
    changes: List[Dict[str, Any]],
    next_design_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministically apply a list of proposed changes to a copy of design.json."""
    mutated = copy.deepcopy(design)
    if next_design_id:
        mutated["design_id"] = next_design_id

    for chg in changes:
        target = chg.get("target", "")
        to_val = chg.get("to")
        if target.startswith("J") and "." in target:
            j_name, field_name = target.split(".", 1)
            if j_name in mutated.get("joints", {}):
                if field_name == "gearbox_id" and (to_val in (None, "null", "None", "")):
                    mutated["joints"][j_name][field_name] = None
                else:
                    mutated["joints"][j_name][field_name] = to_val
        elif target.startswith("links."):
            _, field_name = target.split(".", 1)
            if field_name in ("upper_arm_m", "forearm_m"):
                mutated["links"][field_name] = float(to_val)
            elif field_name == "tube_profile":
                mutated["links"][field_name] = str(to_val)

    return mutated


def _collect_part_citations_and_numbers(part: Dict[str, Any]) -> Tuple[List[Tuple[str, int]], Dict[str, float]]:
    """Return valid (doc, page) pairs and numeric spec values for a catalog part."""
    citations: List[Tuple[str, int]] = []
    numbers: Dict[str, float] = {}

    for key in (
        "stall_torque_nm",
        "continuous_torque_nm",
        "no_load_speed_rad_s",
        "mass_kg",
        "ratio",
        "efficiency",
        "rated_output_torque_nm",
    ):
        field_obj = part.get(key)
        if isinstance(field_obj, dict):
            val = field_obj.get("value")
            if isinstance(val, (int, float)):
                numbers[key] = float(val)
            src = field_obj.get("src") or {}
            doc = src.get("doc")
            page = src.get("page")
            if doc and page is not None:
                citations.append((str(doc).lower(), int(page)))

    price_obj = part.get("price_usd")
    if isinstance(price_obj, dict) and isinstance(price_obj.get("value"), (int, float)):
        numbers["price_usd"] = float(price_obj["value"])

    return citations, numbers


def _check_quoted_numbers_in_text(
    text: str,
    part: Dict[str, Any],
) -> List[str]:
    """Check if explicit spec numbers quoted for a part in rationale/diagnosis are within 2% of catalog."""
    errors: List[str] = []
    if not text:
        return errors

    _, part_numbers = _collect_part_citations_and_numbers(part)
    part_id = part.get("part_id", "")

    # Check explicit patterns like "stall torque 12.5 N·m" or "10.6 N·m stall" or "continuous torque 3.5 N·m"
    patterns = [
        ("stall_torque_nm", r"stall(?:\s+torque)?(?:\s+of|\s+is|\s*:|\s*=)?\s*(\d+(?:\.\d+)?)\s*(?:n[·\.\-\*]?m)"),
        ("stall_torque_nm", r"(\d+(?:\.\d+)?)\s*(?:n[·\.\-\*]?m)\s+stall"),
        ("continuous_torque_nm", r"continuous(?:\s+torque)?(?:\s+of|\s+is|\s*:|\s*=)?\s*(\d+(?:\.\d+)?)\s*(?:n[·\.\-\*]?m)"),
        ("continuous_torque_nm", r"(\d+(?:\.\d+)?)\s*(?:n[·\.\-\*]?m)\s+continuous"),
    ]
    lower_text = text.lower()
    for field_key, pat in patterns:
        if field_key not in part_numbers:
            continue
        true_val = part_numbers[field_key]
        if true_val <= 0:
            continue
        for m in re.finditer(pat, lower_text):
            quoted = float(m.group(1))
            rel_err = abs(quoted - true_val) / true_val
            if rel_err > 0.02:
                errors.append(
                    f"Part '{part_id}' {field_key} quoted as {quoted} in rationale, "
                    f"which differs by >2% from catalog value {true_val}."
                )
    return errors


def validate_proposal(
    proposal: Dict[str, Any],
    current_design: Dict[str, Any],
    catalog: Any,
    materials: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate an Argon proposal against all Section 5.4 rules.

    Returns:
      {
        "valid": bool,
        "errors": List[str],
        "mutated_design": Dict[str, Any] | None,
        "computed_summary": Dict[str, Any] | None
      }
    """
    errors: List[str] = []
    cat_map = index_catalog(catalog)
    mats = materials if materials is not None else load_materials()
    profiles = mats.get("tube_profiles", mats)

    if not isinstance(proposal, dict):
        return {
            "valid": False,
            "errors": ["Proposal must be a JSON object."],
            "mutated_design": None,
            "computed_summary": None,
        }

    changes = proposal.get("changes")
    if not isinstance(changes, list) or len(changes) == 0:
        errors.append("Proposal must include a non-empty 'changes' list.")
        return {
            "valid": False,
            "errors": errors,
            "mutated_design": None,
            "computed_summary": None,
        }

    # Rule 3: At most 3 changes are proposed
    if len(changes) > 3:
        errors.append(f"Too many changes proposed ({len(changes)}); at most 3 changes are allowed per iteration.")

    constraints = current_design.get("constraints", {})
    link_bounds = constraints.get(
        "link_length_bounds_m",
        {"upper_arm_m": [0.22, 0.40], "forearm_m": [0.18, 0.35]},
    )

    # All valid (doc, page) pairs across the entire catalog
    all_catalog_citations = set()
    for p in cat_map.values():
        cits, _ = _collect_part_citations_and_numbers(p)
        for doc_l, pg in cits:
            all_catalog_citations.add((doc_l, pg))

    for idx, chg in enumerate(changes):
        if not isinstance(chg, dict):
            errors.append(f"Change #{idx + 1} is not an object.")
            continue

        target = chg.get("target", "")
        to_val = chg.get("to")
        rationale = str(chg.get("rationale", ""))
        citations = chg.get("citations") or []

        if target not in VALID_TARGETS:
            errors.append(f"Change #{idx + 1} has invalid target '{target}'. Valid targets: {sorted(VALID_TARGETS)}.")
            continue

        # Rule 1: Actuator / gearbox existence and type check
        if target.endswith(".actuator_id"):
            part_id = str(to_val or "")
            if part_id not in cat_map:
                errors.append(f"Change #{idx + 1} proposes actuator '{part_id}' on {target}, which does not exist in catalog.json.")
            else:
                part = cat_map[part_id]
                if part.get("type") != "actuator":
                    errors.append(f"Change #{idx + 1} proposes '{part_id}' on {target}, which has type '{part.get('type')}' instead of 'actuator'.")
                else:
                    errors.extend(_check_quoted_numbers_in_text(rationale, part))

        elif target.endswith(".gearbox_id"):
            if to_val not in (None, "null", "None", ""):
                part_id = str(to_val)
                if part_id not in cat_map:
                    errors.append(f"Change #{idx + 1} proposes gearbox '{part_id}' on {target}, which does not exist in catalog.json.")
                else:
                    part = cat_map[part_id]
                    if part.get("type") != "gearbox":
                        errors.append(f"Change #{idx + 1} proposes '{part_id}' on {target}, which has type '{part.get('type')}' instead of 'gearbox'.")
                    else:
                        errors.extend(_check_quoted_numbers_in_text(rationale, part))

        # Rule 2: Link length bounds & tube profile existence
        elif target in ("links.upper_arm_m", "links.forearm_m"):
            _, link_key = target.split(".", 1)
            try:
                val_f = float(to_val)
                lo, hi = link_bounds.get(link_key, [0.15, 0.50])
                if not (float(lo) - 1e-6 <= val_f <= float(hi) + 1e-6):
                    errors.append(
                        f"Change #{idx + 1} proposes {target}={val_f} m, outside allowed bounds [{lo}, {hi}] m."
                    )
            except (TypeError, ValueError):
                errors.append(f"Change #{idx + 1} proposes non-numeric value '{to_val}' for {target}.")

        elif target == "links.tube_profile":
            prof_id = str(to_val or "")
            if prof_id not in profiles:
                errors.append(
                    f"Change #{idx + 1} proposes tube_profile '{prof_id}', which is not in materials.json ({list(profiles.keys())})."
                )

        # Rule 4: Citations check
        if target.endswith(".actuator_id") or (target.endswith(".gearbox_id") and to_val not in (None, "null", "None", "")):
            if not isinstance(citations, list) or len(citations) == 0:
                errors.append(f"Change #{idx + 1} ({target} -> {to_val}) is missing datasheet citations.")
            else:
                part_id = str(to_val)
                part = cat_map.get(part_id)
                part_cits, _ = _collect_part_citations_and_numbers(part) if part else ([], {})
                part_docs = {d for d, _ in part_cits}
                for cit in citations:
                    if not isinstance(cit, dict):
                        errors.append(f"Change #{idx + 1} has malformed citation entry.")
                        continue
                    c_doc = str(cit.get("doc") or "").strip().lower()
                    c_page = cit.get("page")
                    if not c_doc or c_page is None:
                        errors.append(f"Change #{idx + 1} citation must include 'doc' and 'page'.")
                        continue
                    try:
                        c_page_int = int(c_page)
                    except (TypeError, ValueError):
                        errors.append(f"Change #{idx + 1} citation page '{c_page}' is not an integer.")
                        continue
                    if (c_doc, c_page_int) not in all_catalog_citations:
                        errors.append(
                            f"Change #{idx + 1} cites ({cit.get('doc')}, page {c_page_int}), which does not exist in catalog.json."
                        )
                    elif part is not None:
                        field_name = str(cit.get("field") or "")
                        field_src = (part.get(field_name) or {}).get("src", {}) if isinstance(part.get(field_name), dict) else {}
                        exact = (str(field_src.get("doc", "")).lower(), field_src.get("page"))
                        if exact != (c_doc, c_page_int):
                            errors.append(
                                f"Change #{idx + 1} citation must match part '{part_id}' field '{field_name}' exactly; expected {exact}."
                            )

    if errors:
        return {
            "valid": False,
            "errors": errors,
            "mutated_design": None,
            "computed_summary": None,
        }

    # Rule 5: Deterministic mass, cost, and reach check on mutated design
    try:
        mutated_design = apply_changes_to_design(current_design, changes)
        summary = compute_mass_and_cost(mutated_design, cat_map, mats)
    except Exception as exc:
        errors.append(f"Failed to compute mass/cost for proposed design: {exc}")
        return {
            "valid": False,
            "errors": errors,
            "mutated_design": None,
            "computed_summary": None,
        }

    voltages = {float(cat_map[cfg["actuator_id"]]["rated_voltage_v"])
                for cfg in mutated_design["joints"].values()
                if cat_map[cfg["actuator_id"]].get("rated_voltage_v") is not None}
    if len(voltages) > 1 and not mutated_design.get("allow_mixed_voltage_buses", False):
        errors.append("Selected actuators require different voltage buses; explicitly model separate power buses before combining them.")

    max_mass = float(constraints.get("max_total_mass_kg", 4.0))
    max_cost = float(constraints.get("max_cost_usd", 2500.0))
    min_reach = float(constraints.get("min_reach_m", 0.55))

    if summary["total_mass_kg"] > max_mass:
        errors.append(
            f"Proposed design total_mass_kg ({summary['total_mass_kg']:.3f} kg) exceeds max_total_mass_kg ({max_mass:.3f} kg)."
        )
    if summary["total_cost_usd"] > max_cost:
        errors.append(
            f"Proposed design total_cost_usd (${summary['total_cost_usd']:.2f}) exceeds max_cost_usd (${max_cost:.2f})."
        )
    if summary["reach_m"] < min_reach - 1e-6:
        errors.append(
            f"Proposed design reach_m ({summary['reach_m']:.3f} m) is below min_reach_m ({min_reach:.3f} m)."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "mutated_design": mutated_design if len(errors) == 0 else None,
        "computed_summary": summary,
    }
