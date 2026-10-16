"""Build parametric 4-DOF MJCF models and compute actuator/link mass & cost budgets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
DATA_DIR = Path(__file__).resolve().parent / "data"

_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)

GRAVITY_M_S2 = 9.81
WRIST_LINK_MASS_KG = 0.050
WRIST_LINK_R_OUTER_M = 0.012
WRIST_LINK_R_INNER_M = 0.009
STRUCTURAL_MOUNT_OVERHEAD_USD = 85.0
BASE_STRUCTURE_MASS_KG = 0.220


def load_materials() -> dict[str, Any]:
    with open(DATA_DIR / "materials.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_default_materials() -> dict[str, Any]:
    return load_materials()


def load_default_catalog() -> dict[str, Any]:
    with open(DATA_DIR / "demo" / "catalog.json", "r", encoding="utf-8") as f:
        return json.load(f)


def index_catalog(catalog: Any) -> dict[str, dict[str, Any]]:
    """Return a dict mapping part_id -> part object from a catalog dict or list."""
    if isinstance(catalog, dict):
        if "parts" in catalog and isinstance(catalog["parts"], list):
            return {p["part_id"]: p for p in catalog["parts"] if isinstance(p, dict) and "part_id" in p}
        return {k: v for k, v in catalog.items() if isinstance(v, dict) and "part_id" in v}
    if isinstance(catalog, list):
        return {p["part_id"]: p for p in catalog if isinstance(p, dict) and "part_id" in p}
    return {}


def find_catalog_part(part_id: str | None, catalog: Any) -> dict[str, Any] | None:
    if not part_id:
        return None
    cat_map = index_catalog(catalog)
    return cat_map.get(part_id)



def _val(field: Any, default: float = 0.0) -> float:
    if isinstance(field, dict):
        return float(field.get("value", default))
    if field is None:
        return default
    return float(field)


def get_joint_actuator_specs(joint_cfg: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    """Return effective output stall torque, continuous torque, no-load speed, mass, and price for a joint."""
    act_id = joint_cfg.get("actuator_id")
    gb_id = joint_cfg.get("gearbox_id")

    act = find_catalog_part(act_id, catalog)
    if act is None or act.get("type") != "actuator":
        raise ValueError(f"Unknown or invalid actuator_id: {act_id}")

    tau_s_motor = _val(act.get("stall_torque_nm"))
    tau_cont_motor = _val(act.get("continuous_torque_nm"))
    omega_0_motor = _val(act.get("no_load_speed_rad_s"))
    mass_kg = _val(act.get("mass_kg"))
    price_usd = _val(act.get("price_usd"))

    for field, value in (("stall_torque_nm", tau_s_motor), ("continuous_torque_nm", tau_cont_motor),
                         ("no_load_speed_rad_s", omega_0_motor), ("mass_kg", mass_kg)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Actuator {act_id} requires a finite positive {field}; review catalog extraction.")
    if not isinstance(act.get("price_usd"), dict) or act["price_usd"].get("value") is None or not math.isfinite(price_usd) or price_usd < 0:
        raise ValueError(f"Actuator {act_id} requires a finite non-negative catalog price.")

    ratio = 1.0
    efficiency = 1.0
    rated_out_max = float("inf")

    if gb_id:
        gb = find_catalog_part(gb_id, catalog)
        if gb is None or gb.get("type") != "gearbox":
            raise ValueError(f"Unknown or invalid gearbox_id: {gb_id}")
        ratio = _val(gb.get("ratio"))
        efficiency = _val(gb.get("efficiency"))
        rated_out_max = _val(gb.get("rated_output_torque_nm"), float("inf"))
        gb_mass = _val(gb.get("mass_kg"))
        gb_price = _val(gb.get("price_usd"))
        if (not all(math.isfinite(x) for x in (ratio, efficiency, rated_out_max, gb_mass, gb_price))
                or ratio < 1 or not (0 < efficiency <= 1) or rated_out_max <= 0 or gb_mass <= 0
                or not isinstance(gb.get("price_usd"), dict) or gb["price_usd"].get("value") is None or gb_price < 0):
            raise ValueError(f"Gearbox {gb_id} has missing or invalid mechanical/price specifications.")
        mass_kg += gb_mass
        price_usd += gb_price

    tau_s = min(tau_s_motor * ratio * efficiency, rated_out_max * 1.5)
    tau_cont = min(tau_cont_motor * ratio * efficiency, rated_out_max)
    omega_0 = omega_0_motor / ratio

    return {
        "actuator_id": act_id,
        "gearbox_id": gb_id,
        "ratio": ratio,
        "efficiency": efficiency,
        "tau_s": float(tau_s),
        "tau_cont": float(tau_cont),
        "omega_0": float(omega_0),
        "tau_stall_eff": float(tau_s),
        "tau_cont_eff": float(tau_cont),
        "tau_rated_eff": float(rated_out_max),
        "omega_nl_eff": float(omega_0),
        "mass_kg": float(mass_kg),
        "price_usd": float(price_usd),
        "dimensions_mm": act.get("dimensions_mm"),
        "continuous_torque_derived": bool((act.get("continuous_torque_nm") or {}).get("derived", False)),
        "catalog_source": act.get("provenance", {}).get("source", "user_catalog_unverified"),
        "vendor": act.get("vendor", "ROBOTIS"),
        "series": act.get("series", "DYNAMIXEL"),
    }


def available_torque(
    omega: float | Any,
    tau_s: float,
    omega_0_or_rated: float,
    omega_0_opt: float | None = None,
) -> Any:
    """Linear torque-speed curve: tau_avail(omega) = min(tau_rated, tau_s * max(0, 1 - |omega| / omega_0)).

    Supports both 3-arg call `available_torque(omega, tau_s, omega_0)` and
    4-arg call `available_torque(omega, tau_s, tau_rated_eff, omega_0)`.
    """
    import numpy as np

    if omega_0_opt is None:
        omega_0 = omega_0_or_rated
        tau_rated = float("inf")
    else:
        tau_rated = omega_0_or_rated
        omega_0 = omega_0_opt

    if omega_0 <= 1e-9:
        return np.zeros_like(omega) if isinstance(omega, np.ndarray) else 0.0
    factor = np.maximum(0.0, 1.0 - np.abs(omega) / omega_0)
    lin = tau_s * factor
    return np.minimum(tau_rated, lin) if isinstance(omega, np.ndarray) else float(min(tau_rated, lin))


def compute_tube_mass(length_m: float, tube_profile: str, materials: dict[str, Any]) -> float:
    """Link tube mass = density * pi * (r_o^2 - r_i^2) * length."""
    if tube_profile not in materials:
        raise ValueError(f"Unknown tube_profile: {tube_profile}")
    mat = materials[tube_profile]
    r_o = float(mat["r_outer_m"])
    r_i = float(mat["r_inner_m"])
    rho = float(mat["density_kg_m3"])
    area = math.pi * (r_o**2 - r_i**2)
    return rho * area * float(length_m)


def _composite_link_inertial(
    m_act: float, m_link: float, length_m: float, r_o: float, r_i: float
) -> tuple[float, float, float, float, float]:
    """Combine point-mass actuator at x=0 and hollow cylindrical tube along x in [0, L]."""
    m_total = max(1e-5, m_act + m_link)
    com_x = (m_link * (0.5 * length_m)) / m_total

    # Hollow cylinder inertia about its own CoM (x_c = L/2):
    ixx_link = 0.5 * m_link * (r_o**2 + r_i**2)
    iyy_link = (1.0 / 12.0) * m_link * (3.0 * (r_o**2 + r_i**2) + length_m**2)

    # Parallel axis theorem to composite CoM (com_x)
    d_act = com_x - 0.0
    d_link = (0.5 * length_m) - com_x

    ixx = max(1e-5, ixx_link + 1e-5)
    iyy = max(1e-5, iyy_link + m_link * (d_link**2) + m_act * (d_act**2) + 1e-5)
    izz = iyy
    return m_total, com_x, ixx, iyy, izz


def compute_design_mass_and_cost(
    design: dict[str, Any], catalog: dict[str, Any], materials: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Compute total arm mass (excluding payload), total cost USD, reach, and per-joint breakdown."""
    if materials is None:
        materials = load_materials()

    links = design["links"]
    L1 = float(links["upper_arm_m"])
    L2 = float(links["forearm_m"])
    L3 = float(links["wrist_to_tool_m"])
    base_h = float(links["base_height_m"])
    profile = links["tube_profile"]
    mat = materials[profile]

    m_link1 = compute_tube_mass(L1, profile, materials)
    m_link2 = compute_tube_mass(L2, profile, materials)
    m_link3 = WRIST_LINK_MASS_KG

    joint_specs = {}
    act_mass_sum = 0.0
    act_cost_sum = 0.0
    for j_name in ("J1", "J2", "J3", "J4"):
        spec = get_joint_actuator_specs(design["joints"][j_name], catalog)
        joint_specs[j_name] = spec
        act_mass_sum += spec["mass_kg"]
        act_cost_sum += spec["price_usd"]

    tube_length_total = base_h + L1 + L2 + L3
    tube_cost = tube_length_total * float(mat.get("cost_usd_per_m", 35.0)) + STRUCTURAL_MOUNT_OVERHEAD_USD

    total_mass_kg = BASE_STRUCTURE_MASS_KG + m_link1 + m_link2 + m_link3 + act_mass_sum
    total_cost_usd = act_cost_sum + tube_cost
    reach_m = L1 + L2 + L3

    return {
        "mass_kg": round(total_mass_kg, 4),
        "cost_usd": round(total_cost_usd, 2),
        "total_mass_kg": round(total_mass_kg, 4),
        "total_cost_usd": round(total_cost_usd, 2),
        "reach_m": round(reach_m, 4),
        "m_link1": m_link1,
        "m_link2": m_link2,
        "m_link3": m_link3,
        "joint_specs": joint_specs,
    }


compute_mass_and_cost = compute_design_mass_and_cost



def compute_analytical_j2_static_torque(
    design: dict[str, Any], catalog: dict[str, Any], materials: dict[str, Any] | None = None
) -> float:
    """Hand calculation for static gravity torque at J2 when arm is horizontal (q=0): g * sum(m_i * x_i)."""
    if materials is None:
        materials = load_materials()

    summary = compute_design_mass_and_cost(design, catalog, materials)
    links = design["links"]
    L1 = float(links["upper_arm_m"])
    L2 = float(links["forearm_m"])
    L3 = float(links["wrist_to_tool_m"])
    payload = float(design.get("payload_kg", 0.0))

    m_link1 = summary["m_link1"]
    m_link2 = summary["m_link2"]
    m_link3 = summary["m_link3"]
    m_act_j3 = summary["joint_specs"]["J3"]["mass_kg"]
    m_act_j4 = summary["joint_specs"]["J4"]["mass_kg"]

    moment = (
        m_link1 * (0.5 * L1)
        + m_act_j3 * L1
        + m_link2 * (L1 + 0.5 * L2)
        + m_act_j4 * (L1 + L2)
        + m_link3 * (L1 + L2 + 0.5 * L3)
        + payload * (L1 + L2 + L3)
    )
    return GRAVITY_M_S2 * moment


def build_mjcf(
    design: dict[str, Any], catalog: dict[str, Any], materials: dict[str, Any] | None = None
) -> str:
    """Render valid MuJoCo MJCF XML string for the given design."""
    if materials is None:
        materials = load_materials()

    summary = compute_design_mass_and_cost(design, catalog, materials)
    links = design["links"]
    L1 = float(links["upper_arm_m"])
    L2 = float(links["forearm_m"])
    L3 = float(links["wrist_to_tool_m"])
    base_h = float(links["base_height_m"])
    profile = links["tube_profile"]
    mat = materials[profile]
    r_o = float(mat["r_outer_m"])
    r_i = float(mat["r_inner_m"])

    js = summary["joint_specs"]

    m_sh, com_sh, ix_sh, iy_sh, iz_sh = _composite_link_inertial(
        js["J2"]["mass_kg"], summary["m_link1"], L1, r_o, r_i
    )
    m_el, com_el, ix_el, iy_el, iz_el = _composite_link_inertial(
        js["J3"]["mass_kg"], summary["m_link2"], L2, r_o, r_i
    )
    m_wr, com_wr, ix_wr, iy_wr, iz_wr = _composite_link_inertial(
        js["J4"]["mass_kg"], summary["m_link3"], L3, WRIST_LINK_R_OUTER_M, WRIST_LINK_R_INNER_M
    )

    template = _env.get_template("arm4.xml.j2")
    joints_ctx = [
        {"name": name, "tau_s": js[name]["tau_s"]}
        for name in ("J1", "J2", "J3", "J4")
    ]

    return template.render(
        design_id=design.get("design_id", "arm"),
        base_h=base_h,
        m_base=BASE_STRUCTURE_MASS_KG + js["J1"]["mass_kg"],
        r_o=r_o,
        L1=L1,
        L2=L2,
        L3=L3,
        m_shoulder_total=m_sh,
        com_shoulder_x=com_sh,
        I_shoulder_x=ix_sh,
        I_shoulder_y=iy_sh,
        I_shoulder_z=iz_sh,
        m_elbow_total=m_el,
        com_elbow_x=com_el,
        I_elbow_x=ix_el,
        I_elbow_y=iy_el,
        I_elbow_z=iz_el,
        m_wrist_total=m_wr,
        com_wrist_x=com_wr,
        I_wrist_x=ix_wr,
        I_wrist_y=iy_wr,
        I_wrist_z=iz_wr,
        payload=max(1e-5, float(design.get("payload_kg", 0.0))),
        joints=joints_ctx,
    )
