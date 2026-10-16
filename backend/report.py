"""Solution report generator (Section 6) with inline SVG plots and number-source verification."""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from jinja2 import Environment, FileSystemLoader, select_autoescape

from backend.gemini import generate_report_prose
from backend.model_builder import index_catalog


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _collect_allowed_numbers(*objects: Any) -> List[float]:
    """Recursively collect all numeric values from dicts/lists for prose verification."""
    nums: List[float] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            nums.append(float(obj))
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)

    for root in objects:
        _walk(root)
    return nums


def verify_prose_numbers(prose: Dict[str, str], allowed_numbers: List[float], rel_tol: float = 0.025) -> bool:
    """Check numeric membership only; this does not establish field/unit semantics.

    Reports render authoritative measurements in deterministic paragraphs and
    tables instead of using this helper to approve model-written claims.
    """
    text = " ".join(str(v) for v in prose.values())
    # Strip known model identifiers like J1, J2, J3, J4, XM430-W350, XM540-W270, PG20-2, etc.
    cleaned = re.sub(r"\b(?:J[1-4]|XM\d+-W\d+|XL\d+-W\d+|XH\d+-W\d+|PH\d+-\d+-S\d+-R|AK\d+-\d+|APEX-PG20-\d+|PG20-\d+|v\d+)\b", "", text)
    matches = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", cleaned)
    for m in matches:
        try:
            val = abs(float(m))
        except ValueError:
            continue
        matched = False
        for ref in allowed_numbers:
            ref_abs = abs(ref)
            if ref_abs == 0.0 and val == 0.0:
                matched = True
                break
            denom = max(ref_abs, 1e-6)
            if abs(val - ref_abs) / denom <= rel_tol or abs(val - ref_abs) <= 0.05:
                matched = True
                break
        if not matched:
            return False
    return True


def _build_joint_torque_svg(
    joint_name: str,
    t_series: List[float],
    initial_trace: Dict[str, Any],
    final_trace: Dict[str, Any],
    initial_jm: Dict[str, Any],
    final_jm: Dict[str, Any],
) -> str:
    """Render an inline SVG comparing required torque, available envelope, and continuous band."""
    width, height = 500, 210
    pad_l, pad_r, pad_t, pad_b = 46, 18, 20, 32
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    j_idx = ["J1", "J2", "J3", "J4"].index(joint_name)

    def _get_joint_series(tr: Dict[str, Any], key: str) -> List[float]:
        raw = tr[key]
        if isinstance(raw, dict):
            return [abs(float(x)) for x in raw[joint_name]]
        return [abs(float(row[j_idx])) for row in raw]

    req_init = _get_joint_series(initial_trace, "tau_req")
    req_final = _get_joint_series(final_trace, "tau_req")
    avail_init = _get_joint_series(initial_trace, "tau_avail")
    avail_final = _get_joint_series(final_trace, "tau_avail")

    cont_init = float(initial_jm["tau_cont_eff_nm"])
    cont_final = float(final_jm["tau_cont_eff_nm"])

    max_y = max(
        max(req_init or [1.0]),
        max(req_final or [1.0]),
        max(avail_final or [1.0]),
        cont_final,
        1.0,
    ) * 1.12
    t_max = max(t_series[-1] if t_series else 1.0, 0.1)

    def sx(t: float) -> float:
        return pad_l + (t / t_max) * plot_w

    def sy(v: float) -> float:
        clamped = max(0.0, min(max_y, v))
        return pad_t + plot_h - (clamped / max_y) * plot_h

    def poly(ts: List[float], vs: List[float]) -> str:
        pts = [f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(ts, vs)]
        return " ".join(pts)

    y_cont_init = sy(cont_init)
    y_cont_final = sy(cont_final)

    grid_lines = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        val = max_y * frac
        y = sy(val)
        grid_lines.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#1e293b" stroke-width="1"/>'
            f'<text x="{pad_l - 6}" y="{y + 3:.1f}" fill="#64748b" font-size="10" text-anchor="end" font-family="monospace">{val:.1f}</text>'
        )

    return f"""<svg viewBox="0 0 {width} {height}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" fill="#0b0f19" rx="6"/>
  {"".join(grid_lines)}
  <!-- Continuous torque bands -->
  <line x1="{pad_l}" y1="{y_cont_init:.1f}" x2="{width - pad_r}" y2="{y_cont_init:.1f}" stroke="#ef4444" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.75"/>
  <line x1="{pad_l}" y1="{y_cont_final:.1f}" x2="{width - pad_r}" y2="{y_cont_final:.1f}" stroke="#10b981" stroke-width="1.2" stroke-dasharray="5,3" opacity="0.85"/>
  <!-- Available torque envelope (final) -->
  <polyline fill="none" stroke="#38bdf8" stroke-width="1.5" stroke-dasharray="2,2" points="{poly(t_series, avail_final)}" opacity="0.85"/>
  <!-- Available torque envelope (initial) -->
  <polyline fill="none" stroke="#f97316" stroke-width="1.2" stroke-dasharray="2,2" points="{poly(t_series, avail_init)}" opacity="0.65"/>
  <!-- Required torque (final) -->
  <polyline fill="none" stroke="#10b981" stroke-width="2.0" points="{poly(t_series, req_final)}"/>
  <text x="{pad_l + 6}" y="{pad_t + 12}" fill="#cbd5e1" font-size="10" font-family="sans-serif">
    |&#964;_req| (green) &middot; Final &#964;_avail(&#969;) (cyan) &middot; Final &#964;_cont={cont_final:.2f} N&middot;m (dashed green) &middot; Init &#964;_cont={cont_init:.2f} N&middot;m (red)
  </text>
  <text x="{width / 2:.1f}" y="{height - 6}" fill="#94a3b8" font-size="10" text-anchor="middle" font-family="sans-serif">Time (s) [0.0 &ndash; {t_max:.1f} s] &middot; Y-axis: Torque (N&middot;m)</text>
</svg>"""


def _build_ee_error_svg(
    t_series: List[float],
    initial_err_mm: List[float],
    final_err_mm: List[float],
    threshold_mm: float,
    worst_t_s: float,
) -> str:
    """Render an inline SVG comparing end-effector tracking error before vs. after."""
    width, height = 1020, 220
    pad_l, pad_r, pad_t, pad_b = 50, 20, 22, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    max_y = max(max(initial_err_mm or [threshold_mm]), max(final_err_mm or [threshold_mm]), threshold_mm * 1.5) * 1.08
    t_max = max(t_series[-1] if t_series else 1.0, 0.1)

    def sx(t: float) -> float:
        return pad_l + (t / t_max) * plot_w

    def sy(v: float) -> float:
        clamped = max(0.0, min(max_y, v))
        return pad_t + plot_h - (clamped / max_y) * plot_h

    def poly(ts: List[float], vs: List[float]) -> str:
        return " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(ts, vs))

    y_thresh = sy(threshold_mm)
    x_worst = sx(worst_t_s)

    grid_lines = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        val = max_y * frac
        y = sy(val)
        grid_lines.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#1e293b" stroke-width="1"/>'
            f'<text x="{pad_l - 6}" y="{y + 3:.1f}" fill="#64748b" font-size="10" text-anchor="end" font-family="monospace">{val:.0f}</text>'
        )

    return f"""<svg viewBox="0 0 {width} {height}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" fill="#0b0f19" rx="6"/>
  {"".join(grid_lines)}
  <line x1="{pad_l}" y1="{y_thresh:.1f}" x2="{width - pad_r}" y2="{y_thresh:.1f}" stroke="#38bdf8" stroke-width="1.3" stroke-dasharray="4,4"/>
  <line x1="{x_worst:.1f}" y1="{pad_t}" x2="{x_worst:.1f}" y2="{height - pad_b}" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="2,3"/>
  <polyline fill="none" stroke="#ef4444" stroke-width="2.0" points="{poly(t_series, initial_err_mm)}" opacity="0.9"/>
  <polyline fill="none" stroke="#10b981" stroke-width="2.4" points="{poly(t_series, final_err_mm)}"/>
  <text x="{pad_l + 8}" y="{pad_t + 14}" fill="#cbd5e1" font-size="11" font-family="sans-serif">
    Initial EE Error (red) vs. Selected Candidate EE Error (green) &middot; Threshold = {threshold_mm:.1f} mm (dashed cyan) &middot; Initial Worst Frame t = {worst_t_s:.2f} s
  </text>
  <text x="{width / 2:.1f}" y="{height - 8}" fill="#94a3b8" font-size="10" text-anchor="middle" font-family="sans-serif">Time (s) [0.0 &ndash; {t_max:.1f} s] &middot; Y-axis: End-Effector Position Error ||p_actual &minus; p_target|| (mm)</text>
</svg>"""


def render_solution_report(
    project_id: str,
    initial_design: Dict[str, Any],
    final_design: Dict[str, Any],
    catalog: Dict[str, Any],
    intent: Dict[str, Any],
    motion_summary: Dict[str, Any],
    loop_result: Dict[str, Any],
    approval: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the self-contained HTML engineering verification report."""
    iterations = loop_result["iterations"]
    initial_it = iterations[0]
    final_it = next((it for it in iterations if it.get("design_id") == final_design.get("design_id")), iterations[-1])
    initial_metrics = initial_it["metrics"]
    final_metrics = loop_result.get("final_metrics", final_it["metrics"])
    final_passed = bool(final_it["check"]["passed"])
    initial_passed = bool(initial_it["check"]["passed"])

    mass_delta_kg = round(float(final_metrics["total_mass_kg"]) - float(initial_metrics["total_mass_kg"]), 4)
    cost_delta_usd = round(float(final_metrics["total_cost_usd"]) - float(initial_metrics["total_cost_usd"]), 2)

    prose = generate_report_prose(
        initial_design=initial_design,
        final_design=final_design,
        iterations=iterations,
        intent=intent,
    )

    # Values and pass/fail statements are paired with their exact record. A
    # numerical allow-list cannot verify that an LLM used a number with the
    # correct field, unit or selected design.
    initial_state = "passed" if initial_passed else "failed"
    final_state = "passed" if final_passed else "failed"
    prose["executive_summary"] = (
        f"Initial design {initial_design['design_id']} {initial_state} verification with target tracking RMSE "
        f"{initial_metrics['ee_rmse_mm']:.3f} mm. Selected design {final_design['design_id']} {final_state} "
        f"verification with target tracking RMSE {final_metrics['ee_rmse_mm']:.3f} mm, "
        f"arm mass {final_metrics['total_mass_kg']:.4f} kg and catalog BOM cost ${final_metrics['total_cost_usd']:.2f}."
    )
    if final_metrics.get("orientation_mode") in ("keep_level", "keep_upright"):
        prose["executive_summary"] += (
            f" Constrained tool pitch maximum deviation is {final_metrics['tool_pitch_max_error_deg']:.3f} degrees "
            f"against a {float(final_design['thresholds'].get('tool_pitch_max_error_deg', 5.0)):.3f} degree limit."
        )
    source_info = catalog.get("provenance", {})
    if source_info.get("source") == "project_generated_reference" or str(catalog.get("catalog_id", "")).startswith("demo-"):
        prose["recommendation_notes"] += (
            " Bundled PDFs are project-generated reference documents, not manufacturer original datasheets. "
            "Catalog specifications, thermal estimates, prices and mounting dimensions require source review."
        )

    # Build SVG plots
    initial_trace = loop_result["initial_trace"]
    final_trace = loop_result["final_trace"]
    t_series = initial_trace["t"]

    svg_plots = []
    for j_name in ["J1", "J2", "J3", "J4"]:
        svg_plots.append(
            {
                "title": f"{j_name} Required vs. Available Torque Envelope & Continuous Limit",
                "svg": _build_joint_torque_svg(
                    joint_name=j_name,
                    t_series=t_series,
                    initial_trace=initial_trace,
                    final_trace=final_trace,
                    initial_jm=initial_metrics["joints"][j_name],
                    final_jm=final_metrics["joints"][j_name],
                ),
            }
        )

    def _get_ee_err_mm(tr: Dict[str, Any]) -> List[float]:
        if "ee_error_mm" in tr:
            return [float(x) for x in tr["ee_error_mm"]]
        out: List[float] = []
        for p_des, p_act in zip(tr["ee_des"], tr["ee_act"]):
            dx = float(p_act[0]) - float(p_des[0])
            dy = float(p_act[1]) - float(p_des[1])
            dz = float(p_act[2]) - float(p_des[2])
            out.append(((dx * dx + dy * dy + dz * dz) ** 0.5) * 1000.0)
        return out

    svg_plots.append(
        {
            "title": "End-Effector Cartesian Tracking Error ||p_actual(t) - p_target(t)|| (mm)",
            "svg": _build_ee_error_svg(
                t_series=t_series,
                initial_err_mm=_get_ee_err_mm(initial_trace),
                final_err_mm=_get_ee_err_mm(final_trace),
                threshold_mm=float(final_design["thresholds"]["ee_rmse_mm"]),
                worst_t_s=float(initial_metrics.get("t_worst_ee_s", initial_metrics.get("worst_ee_t_s", 0.0))),
            ),
        }
    )

    # Collect unique parts referenced in initial, iterations, and final designs
    cat_idx = index_catalog(catalog)
    referenced_ids: List[str] = []
    for d_obj in [initial_design, final_design]:
        for j_name in ["J1", "J2", "J3", "J4"]:
            act_id = d_obj["joints"][j_name].get("actuator_id")
            gb_id = d_obj["joints"][j_name].get("gearbox_id")
            if act_id and act_id not in referenced_ids:
                referenced_ids.append(act_id)
            if gb_id and gb_id not in referenced_ids:
                referenced_ids.append(gb_id)
    for it in iterations:
        prop = it.get("proposal") or {}
        for chg in prop.get("changes", []):
            for cand in [chg.get("from"), chg.get("to")]:
                if isinstance(cand, str) and cand in cat_idx and cand not in referenced_ids:
                    referenced_ids.append(cand)

    citations_index = []
    for pid in referenced_ids:
        part = cat_idx.get(pid)
        if not part:
            continue
        if part.get("type") == "actuator":
            stall = part["stall_torque_nm"]
            cont = part["continuous_torque_nm"]
            spd = part["no_load_speed_rad_s"]
            mass = part["mass_kg"]
            price = part["price_usd"]
            specs_summary = (
                f"Stall: {stall['value']} N·m, Cont: {cont['value']} N·m, "
                f"Speed: {spd['value']} rad/s, Mass: {mass['value']} kg, Price: ${price['value']}"
            )
            doc = stall["src"]["doc"]
            page = stall["src"].get("page", 1)
            derived = bool(cont.get("derived", False))
        else:
            ratio = part["ratio"]
            eff = part["efficiency"]
            rated = part["rated_output_torque_nm"]
            mass = part["mass_kg"]
            price = part["price_usd"]
            specs_summary = (
                f"Ratio: {ratio['value']}:1, Eff: {eff['value']}, "
                f"Max Torque: {rated['value']} N·m, Mass: {mass['value']} kg, Price: ${price['value']}"
            )
            doc = ratio["src"]["doc"]
            page = ratio["src"].get("page", 1)
            derived = False

        citations_index.append(
            {
                "part_id": pid,
                "vendor": part.get("vendor", "Vendor"),
                "type": part.get("type", "part"),
                "specs_summary": specs_summary,
                "doc": doc,
                "page": page,
                "derived": derived,
                "source_note": part.get("provenance", {}).get("source", source_info.get("source", "user_catalog_unverified")),
            }
        )

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")
    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return template.render(
        project_id=project_id,
        generated_at=generated_at,
        passed=bool(loop_result["passed"]),
        initial_design=initial_design,
        final_design=final_design,
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        mass_delta_kg=mass_delta_kg,
        cost_delta_usd=cost_delta_usd,
        prose=prose,
        intent=intent,
        motion_summary=motion_summary,
        iterations=iterations,
        naive_baseline=loop_result.get("naive_baseline"),
        svg_plots=svg_plots,
        citations_index=citations_index,
        approval=approval,
    )
