"""Deterministic catalog unit normalization, sanity bounds, and price joining."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TORQUE_TO_NM: Dict[str, float] = {
    "n·m": 1.0,
    "nm": 1.0,
    "n-m": 1.0,
    "n*m": 1.0,
    "mn·m": 1e-3,
    "mnm": 1e-3,
    "kgf·cm": 0.0980665,
    "kgf.cm": 0.0980665,
    "kgf-cm": 0.0980665,
    "kg·cm": 0.0980665,
    "kgcm": 0.0980665,
    "oz·in": 0.00706155,
    "oz-in": 0.00706155,
    "oz.in": 0.00706155,
    "gf·cm": 9.80665e-5,
    "gf.cm": 9.80665e-5,
}

SPEED_TO_RAD_S: Dict[str, float] = {
    "rad/s": 1.0,
    "rads/s": 1.0,
    "rpm": 2.0 * math.pi / 60.0,
    "r/min": 2.0 * math.pi / 60.0,
    "rev/min": 2.0 * math.pi / 60.0,
    "deg/s": math.pi / 180.0,
    "°/s": math.pi / 180.0,
    "rps": 2.0 * math.pi,
    "rev/s": 2.0 * math.pi,
}

MASS_TO_KG: Dict[str, float] = {
    "kg": 1.0,
    "g": 1e-3,
    "gram": 1e-3,
    "grams": 1e-3,
    "lb": 0.453592,
    "lbs": 0.453592,
}

LENGTH_TO_M: Dict[str, float] = {
    "m": 1.0,
    "mm": 1e-3,
    "cm": 1e-2,
    "in": 0.0254,
    "inch": 0.0254,
}


def parse_raw_quantity(raw: str) -> Tuple[Optional[float], Optional[str]]:
    """Extract numeric value and unit string from a raw datasheet string like '30 rpm' or '10.6 N·m'."""
    if not raw or not isinstance(raw, str):
        return None, None
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*([a-zA-Z°·\-\*/]+)", raw.strip())
    if not m:
        m_num = re.search(r"([-+]?\d+(?:\.\d+)?)", raw.strip())
        if m_num:
            return float(m_num.group(1)), None
        return None, None
    return float(m.group(1)), m.group(2).strip().lower()


def convert_torque_to_nm(value: float, unit: Optional[str] = None, raw: Optional[str] = None) -> float:
    """Convert a torque value to N·m."""
    if unit is None and raw:
        parsed_val, parsed_unit = parse_raw_quantity(raw)
        if parsed_unit in TORQUE_TO_NM and parsed_val is not None:
            return round(parsed_val * TORQUE_TO_NM[parsed_unit], 4)
    if unit:
        u = unit.strip().lower()
        if u in TORQUE_TO_NM:
            return round(float(value) * TORQUE_TO_NM[u], 4)
    return round(float(value), 4)


def convert_speed_to_rad_s(value: float, unit: Optional[str] = None, raw: Optional[str] = None) -> float:
    """Convert a rotational speed value to rad/s."""
    if unit is None and raw:
        parsed_val, parsed_unit = parse_raw_quantity(raw)
        if parsed_unit in SPEED_TO_RAD_S and parsed_val is not None:
            return round(parsed_val * SPEED_TO_RAD_S[parsed_unit], 4)
    if unit:
        u = unit.strip().lower()
        if u in SPEED_TO_RAD_S:
            return round(float(value) * SPEED_TO_RAD_S[u], 4)
    return round(float(value), 4)


def convert_mass_to_kg(value: float, unit: Optional[str] = None, raw: Optional[str] = None) -> float:
    """Convert a mass value to kg."""
    if unit is None and raw:
        parsed_val, parsed_unit = parse_raw_quantity(raw)
        if parsed_unit in MASS_TO_KG and parsed_val is not None:
            return round(parsed_val * MASS_TO_KG[parsed_unit], 4)
    if unit:
        u = unit.strip().lower()
        if u in MASS_TO_KG:
            return round(float(value) * MASS_TO_KG[u], 4)
    return round(float(value), 4)


def select_preferred_voltage_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select preferred voltage spec row: 12.0V -> 24.0V -> 48.0V -> highest voltage."""
    if not rows:
        return {}
    by_v = {round(float(r.get("rated_voltage_v", 0.0)), 1): r for r in rows}
    for pref in (12.0, 24.0, 48.0):
        if pref in by_v:
            return by_v[pref]
    return max(rows, key=lambda r: float(r.get("rated_voltage_v", 0.0)))


def load_prices_csv(prices_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load prices.csv into a map keyed by part_id with row number citation."""
    prices: Dict[str, Dict[str, Any]] = {}
    if not prices_path.exists():
        return prices
    with open(prices_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=2):  # Row 1 is header
            pid = (row.get("part_id") or "").strip()
            if not pid:
                continue
            try:
                p_val = float(row.get("price_usd", 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid price for '{pid}' in {prices_path.name} row {idx}") from exc
            if not math.isfinite(p_val) or p_val < 0:
                raise ValueError(f"Price for '{pid}' must be finite and non-negative (row {idx})")
            prices[pid] = {
                "value": p_val,
                "src": {"doc": prices_path.name, "row": idx},
            }
    return prices


def normalize_catalog_part(
    part: Dict[str, Any],
    prices_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Normalize a single extracted catalog part (actuator or gearbox), apply sanity bounds, and join price."""
    import copy

    norm = copy.deepcopy(part)
    ptype = norm.get("type", "actuator")
    warnings: List[str] = list(norm.get("warnings", []))
    confidence = norm.get("confidence", "high")

    if ptype == "actuator":
        # 1. Stall torque
        st_obj = norm.get("stall_torque_nm") or {}
        st_val = st_obj.get("value")
        if st_val is not None:
            st_si = convert_torque_to_nm(float(st_val), st_obj.get("unit"), st_obj.get("raw"))
            st_obj["value"] = st_si
            st_obj["unit"] = "N·m"
            norm["stall_torque_nm"] = st_obj
            if not (0.05 <= st_si <= 500.0):
                confidence = "low"
                warnings.append(f"stall_torque_nm ({st_si} N·m) outside sanity range [0.05, 500]")
        else:
            confidence = "low"
            warnings.append("Missing stall_torque_nm")
            st_si = 0.0

        # 2. Continuous torque (derive 0.33 * stall if missing or null)
        ct_obj = norm.get("continuous_torque_nm") or {}
        ct_val = ct_obj.get("value")
        if ct_val is None:
            derived_val = round(0.33 * st_si, 4)
            norm["continuous_torque_nm"] = {
                "value": derived_val,
                "src": dict(st_obj.get("src") or {"doc": "unknown.pdf", "page": 1}),
                "derived": True,
                "unit": "N·m",
                "raw": ct_obj.get("raw") or f"Derived: 0.33 * {st_si} N·m",
            }
            ct_si = derived_val
            confidence = "low"
            warnings.append("Continuous torque is an engineering estimate (0.33 × stall), not a measured thermal rating.")
        else:
            ct_si = convert_torque_to_nm(float(ct_val), ct_obj.get("unit"), ct_obj.get("raw"))
            ct_obj["value"] = ct_si
            ct_obj["unit"] = "N·m"
            ct_obj["derived"] = bool(ct_obj.get("derived", False))
            norm["continuous_torque_nm"] = ct_obj
            if ct_obj["derived"]:
                confidence = "low"
                warnings.append("Continuous torque is derived; thermal capability requires independent review.")

        if st_si > 0 and ct_si > st_si:
            confidence = "low"
            warnings.append(
                f"continuous_torque_nm ({ct_si} N·m) exceeds stall_torque_nm ({st_si} N·m)"
            )

        # 3. No-load speed
        nl_obj = norm.get("no_load_speed_rad_s") or {}
        nl_val = nl_obj.get("value")
        if nl_val is not None:
            nl_si = convert_speed_to_rad_s(float(nl_val), nl_obj.get("unit"), nl_obj.get("raw"))
            nl_obj["value"] = nl_si
            nl_obj["unit"] = "rad/s"
            norm["no_load_speed_rad_s"] = nl_obj
            if not (0.5 <= nl_si <= 100.0):
                confidence = "low"
                warnings.append(f"no_load_speed_rad_s ({nl_si} rad/s) outside sanity range [0.5, 100]")
        else:
            confidence = "low"
            warnings.append("Missing no_load_speed_rad_s")

        # 4. Mass
        m_obj = norm.get("mass_kg") or {}
        m_val = m_obj.get("value")
        if m_val is not None:
            m_si = convert_mass_to_kg(float(m_val), m_obj.get("unit"), m_obj.get("raw"))
            m_obj["value"] = m_si
            m_obj["unit"] = "kg"
            norm["mass_kg"] = m_obj
            if not (0.01 <= m_si <= 20.0):
                confidence = "low"
                warnings.append(f"mass_kg ({m_si} kg) outside sanity range [0.01, 20]")
        else:
            confidence = "low"
            warnings.append("Missing mass_kg")

    elif ptype == "gearbox":
        # 1. Ratio
        r_obj = norm.get("ratio") or {}
        r_val = r_obj.get("value")
        if r_val is not None:
            r_f = float(r_val)
            r_obj["value"] = r_f
            norm["ratio"] = r_obj
            if not (1.0 <= r_f <= 500.0):
                confidence = "low"
                warnings.append(f"ratio ({r_f}) outside sanity range [1, 500]")
        else:
            confidence = "low"
            warnings.append("Missing gearbox ratio")

        # 2. Efficiency
        eff_obj = norm.get("efficiency") or {}
        eff_val = eff_obj.get("value")
        if eff_val is not None:
            eff_f = float(eff_val)
            if eff_f > 1.0 and eff_f <= 100.0:
                eff_f = round(eff_f / 100.0, 4)
            eff_obj["value"] = eff_f
            norm["efficiency"] = eff_obj
            if not (0.3 <= eff_f <= 1.0):
                confidence = "low"
                warnings.append(f"efficiency ({eff_f}) outside sanity range [0.3, 1.0]")
        else:
            confidence = "low"
            warnings.append("Missing gearbox efficiency")

        # 3. Rated output torque
        rot_obj = norm.get("rated_output_torque_nm") or {}
        rot_val = rot_obj.get("value")
        if rot_val is not None:
            rot_si = convert_torque_to_nm(float(rot_val), rot_obj.get("unit"), rot_obj.get("raw"))
            rot_obj["value"] = rot_si
            rot_obj["unit"] = "N·m"
            norm["rated_output_torque_nm"] = rot_obj

        # 4. Mass
        m_obj = norm.get("mass_kg") or {}
        m_val = m_obj.get("value")
        if m_val is not None:
            m_si = convert_mass_to_kg(float(m_val), m_obj.get("unit"), m_obj.get("raw"))
            m_obj["value"] = m_si
            m_obj["unit"] = "kg"
            norm["mass_kg"] = m_obj
            if not (0.01 <= m_si <= 20.0):
                confidence = "low"
                warnings.append(f"mass_kg ({m_si} kg) outside sanity range [0.01, 20]")

    # Price join
    pid = norm.get("part_id", "")
    if prices_map is not None:
        if pid in prices_map:
            norm["price_usd"] = dict(prices_map[pid])
        elif not norm.get("price_usd") or norm["price_usd"].get("value") is None:
            norm["price_usd"] = {"value": None, "src": {"doc": "prices.csv", "row": None}}
            warnings.append(f"Part '{pid}' missing from prices.csv")

    norm["confidence"] = confidence
    if warnings:
        norm["warnings"] = list(dict.fromkeys(warnings))
    norm.setdefault("user_confirmed", False)
    return norm


def normalize_catalog(
    catalog_data: Dict[str, Any],
    prices_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Normalize an entire catalog dictionary."""
    prices_map = load_prices_csv(prices_path) if prices_path else None
    parts = catalog_data.get("parts", [])
    normalized_parts = [normalize_catalog_part(p, prices_map=prices_map) for p in parts]
    result = dict(catalog_data)
    if str(result.get("catalog_id", "")).startswith("demo-"):
        result["provenance"] = {
            "source": "project_generated_reference", "manufacturer_verified": False,
            "requires_spec_review": True,
            "note": "Bundled PDFs were generated by this project; they are not manufacturer originals.",
        }
        for part in normalized_parts:
            part.setdefault("provenance", dict(result["provenance"]))
    result["parts"] = normalized_parts
    return result
