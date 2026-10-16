"""Unit tests for M4: catalog_normalize.py, validate.py, and loop.py."""

from __future__ import annotations

import json
import os
from pathlib import Path

from backend.catalog_normalize import (
    convert_mass_to_kg,
    convert_speed_to_rad_s,
    convert_torque_to_nm,
    normalize_catalog_part,
)
from backend.loop import run_design_loop
from backend.validate import validate_proposal


DEMO_DIR = Path(__file__).resolve().parent.parent / "backend" / "data" / "demo"


def test_validator_rejects_fake_part_id() -> None:
    """Acceptance gate M4: Validator must reject an injected fake part ID."""
    design = json.loads((DEMO_DIR / "design.json").read_text())
    catalog = json.loads((DEMO_DIR / "catalog.json").read_text())

    fake_proposal = {
        "diagnosis": "J2 fails peak and continuous torque during lift.",
        "changes": [
            {
                "target": "J2.actuator_id",
                "from": "XM430-W350",
                "to": "FAKE-MOTOR-9000",
                "rationale": "Swap to invented actuator.",
                "citations": [{"doc": "xm540-w270.pdf", "page": 1, "field": "stall_torque_nm"}],
            }
        ],
        "predicted_effect": {
            "j2_peak_ratio": 0.45,
            "j2_rms_ratio": 0.65,
            "total_mass_kg": 0.85,
            "total_cost_usd": 1200.0,
        },
    }

    res = validate_proposal(fake_proposal, design, catalog)
    assert res["valid"] is False
    assert any("FAKE-MOTOR-9000" in err for err in res["errors"]), f"Expected fake part error, got: {res['errors']}"


def test_validator_rejects_wrong_part_type_and_bad_citation() -> None:
    """Validator must reject putting a gearbox into an actuator slot or citing a fabricated number."""
    design = json.loads((DEMO_DIR / "design.json").read_text())
    catalog = json.loads((DEMO_DIR / "catalog.json").read_text())

    # 1. Gearbox in actuator slot
    bad_type_proposal = {
        "diagnosis": "Test wrong part type.",
        "changes": [
            {
                "target": "J2.actuator_id",
                "from": "XM430-W350",
                "to": "APEX-PG20-2",
                "rationale": "Put gearbox in actuator slot.",
                "citations": [{"doc": "apex-pg20-2.pdf", "page": 1, "field": "ratio"}],
            }
        ],
        "predicted_effect": {
            "j2_peak_ratio": 0.5,
            "j2_rms_ratio": 0.7,
            "total_mass_kg": 0.85,
            "total_cost_usd": 1200.0,
        },
    }
    res1 = validate_proposal(bad_type_proposal, design, catalog)
    assert res1["valid"] is False

    # 2. Quoted stall torque > 2% off from catalog value (10.6 N·m)
    bad_quote_proposal = {
        "diagnosis": "Test fabricated number in rationale.",
        "changes": [
            {
                "target": "J2.actuator_id",
                "from": "XM430-W350",
                "to": "XM540-W270",
                "rationale": "Upgrade to XM540-W270 with stall torque 19.5 N·m.",
                "citations": [{"doc": "xm540-w270.pdf", "page": 1, "field": "stall_torque_nm"}],
            }
        ],
        "predicted_effect": {
            "j2_peak_ratio": 0.45,
            "j2_rms_ratio": 0.65,
            "total_mass_kg": 0.87,
            "total_cost_usd": 1276.15,
        },
    }
    res2 = validate_proposal(bad_quote_proposal, design, catalog)
    assert res2["valid"] is False
    assert any(">2%" in err for err in res2["errors"]), f"Expected >2% quote error, got: {res2['errors']}"


def test_catalog_normalize_units_and_derived_continuous() -> None:
    """Verify deterministic unit conversion and 0.33 * stall derived continuous torque fallback."""
    assert abs(convert_torque_to_nm(10.0, "kgf·cm") - 0.9807) < 1e-3
    assert abs(convert_speed_to_rad_s(30.0, "rpm") - 3.1416) < 1e-3
    assert abs(convert_mass_to_kg(165.0, "g") - 0.165) < 1e-5

    raw_part = {
        "part_id": "TEST-ACT-1",
        "type": "actuator",
        "vendor": "TEST",
        "rated_voltage_v": 12.0,
        "stall_torque_nm": {"value": 6.0, "src": {"doc": "test.pdf", "page": 1}},
        "continuous_torque_nm": {"value": None, "src": {"doc": "test.pdf", "page": 1}},
        "no_load_speed_rad_s": {"value": 30.0, "unit": "rpm", "src": {"doc": "test.pdf", "page": 1}},
        "mass_kg": {"value": 120.0, "unit": "g", "src": {"doc": "test.pdf", "page": 1}},
    }
    norm = normalize_catalog_part(raw_part)
    assert norm["continuous_torque_nm"]["derived"] is True
    assert abs(norm["continuous_torque_nm"]["value"] - 1.98) < 1e-3
    assert abs(norm["no_load_speed_rad_s"]["value"] - 3.1416) < 1e-3
    assert abs(norm["mass_kg"]["value"] - 0.120) < 1e-4


def test_design_loop_reaches_pass_in_at_most_4_iterations() -> None:
    """Acceptance gate M4: Loop reaches passed: true on the demo case in at most 4 iterations."""
    os.environ["MOTIONFIT_OFFLINE"] = "1"
    try:
        design = json.loads((DEMO_DIR / "design.json").read_text())
        catalog = json.loads((DEMO_DIR / "catalog.json").read_text())
        target = json.loads((DEMO_DIR / "target.json").read_text())
        intent = json.loads((DEMO_DIR / "intent.json").read_text())

        events = []
        bundle = run_design_loop(
            initial_design=design,
            target=target,
            catalog=catalog,
            intent=intent,
            max_iterations=5,
            optimize_after_pass=True,
            include_naive_baseline=True,
            on_event=lambda e: events.append(e),
        )

        assert bundle["passed"] is True
        assert bundle["iterations_count"] <= 4
        assert bundle["iterations"][0]["check"]["passed"] is False
        assert bundle["final_metrics"]["ee_rmse_mm"] <= 15.0
        assert bundle["naive_baseline"] is not None
        assert bundle["naive_baseline"]["passed"] is False
        assert any(e["event"] == "loop_completed" for e in events)
    finally:
        os.environ.pop("MOTIONFIT_OFFLINE", None)
