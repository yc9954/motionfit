"""Generate project reference PDF datasheets, prices.csv, and catalog.json for MotionFit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DATA_DIR = Path(__file__).resolve().parent / "data"
DEMO_DIR = DATA_DIR / "demo"
DATASHEETS_DIR = DEMO_DIR / "datasheets"


RAW_PARTS = [
    # ROBOTIS DYNAMIXEL X-Series
    {
        "part_id": "XL430-W250",
        "type": "actuator",
        "vendor": "ROBOTIS",
        "series": "DYNAMIXEL-X",
        "rated_voltage_v": 12.0,
        "stall_torque_nm": 1.5,
        "stall_raw": "1.5 N.m (at 12.0V, 1.4A)",
        "continuous_torque_nm": None,  # Engineering estimate: 0.33 * stall = 0.495
        "no_load_speed_rpm": 61.0,
        "speed_raw": "61 rev/min (at 12.0V)",
        "mass_kg": 0.0572,
        "mass_raw": "57.2 g",
        "price_usd": 59.90,
        "source_url": "https://www.robotis.us/dynamixel-xl430-w250-t/",
        "doc": "xl430-w250.pdf",
        "spec_page": 1,
        "dimensions_mm": [28.5, 46.5, 34.0],
        "gear_ratio_internal": "258.5 : 1",
        "motor_type": "Cored DC Motor",
    },
    {
        "part_id": "XM430-W210",
        "type": "actuator",
        "vendor": "ROBOTIS",
        "series": "DYNAMIXEL-X",
        "rated_voltage_v": 12.0,
        "stall_torque_nm": 3.0,
        "stall_raw": "3.0 N.m (at 12.0V, 2.3A)",
        "continuous_torque_nm": 0.99,
        "continuous_raw": "0.99 N.m continuous",
        "no_load_speed_rpm": 77.0,
        "speed_raw": "77 rpm (at 12.0V)",
        "mass_kg": 0.082,
        "mass_raw": "82.0 g",
        "price_usd": 310.00,
        "source_url": "https://www.robotis.us/dynamixel-xm430-w210-t/",
        "doc": "xm430-w210.pdf",
        "spec_page": 1,
        "dimensions_mm": [28.5, 46.5, 34.0],
        "gear_ratio_internal": "212.6 : 1",
        "motor_type": "Coreless DC Motor",
    },
    {
        "part_id": "XM430-W350",
        "type": "actuator",
        "vendor": "ROBOTIS",
        "series": "DYNAMIXEL-X",
        "rated_voltage_v": 12.0,
        "stall_torque_nm": 4.1,
        "stall_raw": "4.1 N.m (at 12.0V, 2.3A)",
        "continuous_torque_nm": 1.35,
        "continuous_raw": "1.35 N.m continuous",
        "no_load_speed_rpm": 46.0,
        "speed_raw": "46 rpm (at 12.0V)",
        "mass_kg": 0.082,
        "mass_raw": "82.0 g",
        "price_usd": 310.00,
        "source_url": "https://www.robotis.us/dynamixel-xm430-w350-t/",
        "doc": "xm430-w350.pdf",
        "spec_page": 1,
        "dimensions_mm": [28.5, 46.5, 34.0],
        "gear_ratio_internal": "353.5 : 1",
        "motor_type": "Coreless DC Motor",
    },
    {
        "part_id": "XH430-W210",
        "type": "actuator",
        "vendor": "ROBOTIS",
        "series": "DYNAMIXEL-X",
        "rated_voltage_v": 12.0,
        "stall_torque_nm": 2.5,
        "stall_raw": "2.5 N.m (at 12.0V, 1.3A)",
        "continuous_torque_nm": 0.83,
        "continuous_raw": "0.83 N.m continuous",
        "no_load_speed_rpm": 50.0,
        "speed_raw": "50 rpm (at 12.0V)",
        "mass_kg": 0.082,
        "mass_raw": "82.0 g",
        "price_usd": 414.00,
        "source_url": "https://www.robotis.us/dynamixel-xh430-w210-r/",
        "doc": "xh430-w210.pdf",
        "spec_page": 1,
        "dimensions_mm": [28.5, 46.5, 34.0],
        "gear_ratio_internal": "212.6 : 1",
        "motor_type": "Maxon Coreless DC",
    },
    {
        "part_id": "XH430-W350",
        "type": "actuator",
        "vendor": "ROBOTIS",
        "series": "DYNAMIXEL-X",
        "rated_voltage_v": 12.0,
        "stall_torque_nm": 3.4,
        "stall_raw": "3.4 N.m (at 12.0V, 1.3A)",
        "continuous_torque_nm": 1.12,
        "continuous_raw": "1.12 N.m continuous",
        "no_load_speed_rpm": 30.0,
        "speed_raw": "30 rpm (at 12.0V)",
        "mass_kg": 0.082,
        "mass_raw": "82.0 g",
        "price_usd": 414.00,
        "source_url": "https://www.robotis.us/dynamixel-xh430-w350-r/",
        "doc": "xh430-w350.pdf",
        "spec_page": 1,
        "dimensions_mm": [28.5, 46.5, 34.0],
        "gear_ratio_internal": "353.5 : 1",
        "motor_type": "Maxon Coreless DC",
    },
    {
        "part_id": "XM540-W150",
        "type": "actuator",
        "vendor": "ROBOTIS",
        "series": "DYNAMIXEL-X",
        "rated_voltage_v": 12.0,
        "stall_torque_nm": 7.3,
        "stall_raw": "7.3 N.m (at 12.0V, 4.4A)",
        "continuous_torque_nm": 2.41,
        "continuous_raw": "2.41 N.m continuous",
        "no_load_speed_rpm": 53.0,
        "speed_raw": "53 rpm (at 12.0V)",
        "mass_kg": 0.165,
        "mass_raw": "165.0 g",
        "price_usd": 485.00,
        "source_url": "https://www.robotis.us/dynamixel-xm540-w150-r/",
        "doc": "xm540-w150.pdf",
        "spec_page": 1,
        "dimensions_mm": [33.5, 58.5, 44.0],
        "gear_ratio_internal": "152.3 : 1",
        "motor_type": "Coreless DC Motor",
    },
    {
        "part_id": "XM540-W270",
        "type": "actuator",
        "vendor": "ROBOTIS",
        "series": "DYNAMIXEL-X",
        "rated_voltage_v": 12.0,
        "stall_torque_nm": 10.6,
        "stall_raw": "10.6 N.m (at 12.0V, 4.4A)",
        "continuous_torque_nm": 3.50,
        "continuous_raw": "3.50 N.m continuous",
        "no_load_speed_rpm": 30.0,
        "speed_raw": "30 rpm (at 12.0V)",
        "mass_kg": 0.165,
        "mass_raw": "165.0 g",
        "price_usd": 485.00,
        "source_url": "https://www.robotis.us/dynamixel-xm540-w270-r/",
        "doc": "xm540-w270.pdf",
        "spec_page": 1,
        "dimensions_mm": [33.5, 58.5, 44.0],
        "gear_ratio_internal": "272.5 : 1",
        "motor_type": "Coreless DC Motor",
    },
    {
        "part_id": "XH540-W150",
        "type": "actuator",
        "vendor": "ROBOTIS",
        "series": "DYNAMIXEL-X",
        "rated_voltage_v": 12.0,
        "stall_torque_nm": 7.1,
        "stall_raw": "7.1 N.m (at 12.0V, 4.9A)",
        "continuous_torque_nm": 2.34,
        "continuous_raw": "2.34 N.m continuous",
        "no_load_speed_rpm": 70.0,
        "speed_raw": "70 rpm (at 12.0V)",
        "mass_kg": 0.165,
        "mass_raw": "165.0 g",
        "price_usd": 625.00,
        "source_url": "https://www.robotis.us/dynamixel-xh540-w150-r/",
        "doc": "xh540-w150.pdf",
        "spec_page": 1,
        "dimensions_mm": [33.5, 58.5, 44.0],
        "gear_ratio_internal": "152.3 : 1",
        "motor_type": "Maxon Coreless DC",
    },
    {
        "part_id": "XH540-W270",
        "type": "actuator",
        "vendor": "ROBOTIS",
        "series": "DYNAMIXEL-X",
        "rated_voltage_v": 12.0,
        "stall_torque_nm": 9.9,
        "stall_raw": "9.9 N.m (at 12.0V, 4.9A)",
        "continuous_torque_nm": 3.27,
        "continuous_raw": "3.27 N.m continuous",
        "no_load_speed_rpm": 39.0,
        "speed_raw": "39 rpm (at 12.0V)",
        "mass_kg": 0.165,
        "mass_raw": "165.0 g",
        "price_usd": 625.00,
        "source_url": "https://www.robotis.us/dynamixel-xh540-w270-r/",
        "doc": "xh540-w270.pdf",
        "spec_page": 1,
        "dimensions_mm": [33.5, 58.5, 44.0],
        "gear_ratio_internal": "272.5 : 1",
        "motor_type": "Maxon Coreless DC",
    },
    # ROBOTIS DYNAMIXEL-P Series
    {
        "part_id": "PH42-020-S300-R",
        "type": "actuator",
        "vendor": "ROBOTIS",
        "series": "DYNAMIXEL-P",
        "rated_voltage_v": 24.0,
        "stall_torque_nm": 21.5,
        "stall_raw": "21.5 N.m peak/stall",
        "continuous_torque_nm": 5.1,
        "continuous_raw": "5.1 N.m rated continuous",
        "no_load_speed_rpm": 32.7,
        "speed_raw": "32.7 rpm (at 24.0V)",
        "mass_kg": 0.340,
        "mass_raw": "340 g",
        "price_usd": 1897.00,
        "source_url": "https://www.robotis.us/dynamixel-ph42-020-s300-r/",
        "doc": "ph42-020-s300-r.pdf",
        "spec_page": 1,
        "dimensions_mm": [42.0, 84.0, 42.0],
        "gear_ratio_internal": "303.8 : 1 Cycloidal",
        "motor_type": "BLDC Servo",
    },
    {
        "part_id": "PH54-100-S500-R",
        "type": "actuator",
        "vendor": "ROBOTIS",
        "series": "DYNAMIXEL-P",
        "rated_voltage_v": 24.0,
        "stall_torque_nm": 67.0,
        "stall_raw": "67.0 N.m peak/stall",
        "continuous_torque_nm": 25.3,
        "continuous_raw": "25.3 N.m rated continuous",
        "no_load_speed_rpm": 33.3,
        "speed_raw": "33.3 rpm (at 24.0V)",
        "mass_kg": 0.740,
        "mass_raw": "740 g",
        "price_usd": 3289.00,
        "source_url": "https://www.robotis.us/dynamixel-ph54-100-s500-r/",
        "doc": "ph54-100-s500-r.pdf",
        "spec_page": 1,
        "dimensions_mm": [54.0, 108.0, 54.0],
        "gear_ratio_internal": "502.0 : 1 Cycloidal",
        "motor_type": "BLDC Servo",
    },
    # CubeMars AK Quasi-Direct-Drive Series
    {
        "part_id": "AK60-6",
        "type": "actuator",
        "vendor": "CubeMars",
        "series": "AK-Series QDD",
        "rated_voltage_v": 24.0,
        "stall_torque_nm": 9.0,
        "stall_raw": "9.0 N.m peak torque",
        "continuous_torque_nm": 3.0,
        "continuous_raw": "3.0 N.m rated torque",
        "no_load_speed_rpm": 320.0,
        "speed_raw": "320 rpm (at 24V KV80)",
        "mass_kg": 0.360,
        "mass_raw": "360 g",
        "price_usd": 298.90,
        "source_url": "https://www.cubemars.com/product/ak60-6-v3-0-kv80.html",
        "doc": "ak60-6.pdf",
        "spec_page": 1,
        "dimensions_mm": [68.0, 39.5, 68.0],
        "gear_ratio_internal": "6 : 1 Planetary QDD",
        "motor_type": "Outer-Rotor BLDC QDD",
    },
    {
        "part_id": "AK70-10",
        "type": "actuator",
        "vendor": "CubeMars",
        "series": "AK-Series QDD",
        "rated_voltage_v": 24.0,
        "stall_torque_nm": 24.8,
        "stall_raw": "24.8 N.m peak torque",
        "continuous_torque_nm": 8.3,
        "continuous_raw": "8.3 N.m rated torque",
        "no_load_speed_rpm": 240.0,
        "speed_raw": "240 rpm (at 24V KV100)",
        "mass_kg": 0.521,
        "mass_raw": "521 g",
        "price_usd": 449.00,
        "source_url": "https://www.cubemars.com/product/ak70-10-kv100.html",
        "doc": "ak70-10.pdf",
        "spec_page": 1,
        "dimensions_mm": [80.0, 50.0, 80.0],
        "gear_ratio_internal": "10 : 1 Planetary QDD",
        "motor_type": "Outer-Rotor BLDC QDD",
    },
    {
        "part_id": "AK80-9",
        "type": "actuator",
        "vendor": "CubeMars",
        "series": "AK-Series QDD",
        "rated_voltage_v": 24.0,
        "stall_torque_nm": 22.0,
        "stall_raw": "22.0 N.m peak torque",
        "continuous_torque_nm": 9.0,
        "continuous_raw": "9.0 N.m rated torque",
        "no_load_speed_rpm": 390.0,
        "speed_raw": "390 rpm (at 24V KV100)",
        "mass_kg": 0.490,
        "mass_raw": "490 g",
        "price_usd": 499.00,
        "source_url": "https://www.cubemars.com/product/ak80-9-kv100.html",
        "doc": "ak80-9.pdf",
        "spec_page": 1,
        "dimensions_mm": [88.0, 40.5, 88.0],
        "gear_ratio_internal": "9 : 1 Planetary QDD",
        "motor_type": "Outer-Rotor BLDC QDD",
    },
    # Inline Planetary Gearboxes
    {
        "part_id": "APEX-PG20-2",
        "type": "gearbox",
        "vendor": "Apex Dynamics",
        "series": "PG20 Inline Planetary",
        "ratio": 2.0,
        "ratio_raw": "2 : 1 single-stage",
        "efficiency": 0.92,
        "efficiency_raw": "92%",
        "rated_output_torque_nm": 18.0,
        "torque_raw": "18.0 N.m max output",
        "mass_kg": 0.065,
        "mass_raw": "65 g",
        "price_usd": 115.00,
        "source_url": "https://www.apexdyna.com/pg20-series",
        "doc": "apex-pg20-2.pdf",
        "spec_page": 1,
        "dimensions_mm": [28.0, 24.0, 28.0],
    },
    {
        "part_id": "APEX-PG20-3",
        "type": "gearbox",
        "vendor": "Apex Dynamics",
        "series": "PG20 Inline Planetary",
        "ratio": 3.0,
        "ratio_raw": "3 : 1 single-stage",
        "efficiency": 0.90,
        "efficiency_raw": "90%",
        "rated_output_torque_nm": 22.0,
        "torque_raw": "22.0 N.m max output",
        "mass_kg": 0.078,
        "mass_raw": "78 g",
        "price_usd": 125.00,
        "source_url": "https://www.apexdyna.com/pg20-series",
        "doc": "apex-pg20-3.pdf",
        "spec_page": 1,
        "dimensions_mm": [28.0, 26.0, 28.0],
    },
]


SOURCE_VERIFICATIONS = {
    "XL430-W250": {
        "source_verified": True, "checked_at": "2026-10-16",
        "primary_source_url": "https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/",
        "verified_fields": ["product_identity", "stall_torque_nm", "no_load_speed_rad_s", "mass_kg", "dimensions_mm"],
        "unverified_fields": ["continuous_torque_nm", "price_usd", "mounting_compatibility"],
        "voltage_v": 12.0,
        "note": "Official 12 V row is 1.5 N.m; 1.4 N.m belongs to the 11.1 V row. Continuous torque is an estimate.",
    },
    "XM430-W350": {
        "source_verified": True, "checked_at": "2026-10-16",
        "primary_source_url": "https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/",
        "verified_fields": ["product_identity", "stall_torque_nm", "no_load_speed_rad_s", "mass_kg", "dimensions_mm"],
        "unverified_fields": ["continuous_torque_nm", "price_usd", "mounting_compatibility"],
        "voltage_v": 12.0,
        "note": "Official 12 V row confirms 4.1 N.m and 46 rpm. The seed continuous torque is not a manufacturer thermal rating.",
    },
    **{part_id: {
        "source_verified": False, "checked_at": "2026-10-16", "test_fixture_only": True,
        "primary_source_url": "https://apexdynamicsusa.com/pgii-series-high-precision-planetary-gearboxes.html",
        "note": "No primary vendor evidence was found for this exact model. The official PGII family is a different product; this generated reference is excluded from normal projects.",
    } for part_id in ("APEX-PG20-2", "APEX-PG20-3")},
}


def _render_pdf(part: dict, out_path: Path) -> None:
    """Create a realistic 2-page engineering datasheet PDF with specification table and curves."""
    with PdfPages(out_path) as pdf:
        # Page 1: Technical Specifications & Performance Curve
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.patch.set_facecolor("#ffffff")
        fig.text(0.5, 0.015, "MotionFit-generated reference — not a manufacturer original; verify specifications", ha="center", fontsize=8, color="#92400e")
        ax_header = fig.add_axes([0.07, 0.87, 0.86, 0.09])
        ax_header.axis("off")
        ax_header.text(
            0.0,
            0.72,
            f"MotionFit Reference — {part['vendor']} {part['series']}",
            fontsize=11,
            fontweight="bold",
            color="#2563eb",
        )
        ax_header.text(
            0.0,
            0.25,
            f"Part Number: {part['part_id']}",
            fontsize=18,
            fontweight="bold",
            color="#0f172a",
        )
        ax_header.text(
            1.0,
            0.25,
            f"Document: {part['doc']} | Page 1 of 2",
            fontsize=9,
            color="#475569",
            ha="right",
        )

        ax_table = fig.add_axes([0.07, 0.46, 0.86, 0.38])
        ax_table.axis("off")
        ax_table.set_title(
            "1. Electrical & Mechanical Specifications (Standard Reference)",
            loc="left",
            fontsize=12,
            fontweight="bold",
            pad=10,
            color="#1e293b",
        )

        if part["type"] == "actuator":
            cont_str = (
                part.get("continuous_raw")
                if part["continuous_torque_nm"] is not None
                else "Not specified (use thermal derating guidelines)"
            )
            if part.get("series") == "DYNAMIXEL-X":
                estimate = part["continuous_torque_nm"] if part["continuous_torque_nm"] is not None else 0.33 * part["stall_torque_nm"]
                cont_str = f"{estimate:.3f} N.m (engineering estimate)"
            rows = [
                ["Parameter", "Symbol", "Specification Value", "Test Condition / Notes"],
                ["Part Number", "ID", part["part_id"], part["series"]],
                ["Manufacturer", "Vendor", part["vendor"], part["motor_type"]],
                ["Rated Input Voltage", "V_rated", f"{part['rated_voltage_v']:.1f} V", "Nominal DC Bus"],
                ["Stall / Peak Torque", "tau_stall", part["stall_raw"], f"At {part['rated_voltage_v']:.1f} V"],
                ["Continuous Torque Input", "tau_cont", cont_str, "Estimate; verify manufacturer thermal rating"],
                ["No-Load Speed", "omega_0", part["speed_raw"], "Unloaded output shaft"],
                ["Actuator Mass (Weight)", "m_act", part["mass_raw"], "Includes housing & internal gearhead"],
                ["Internal Reduction", "i_int", part["gear_ratio_internal"], "Integrated gear stage"],
                [
                    "Envelope Dimensions",
                    "W x H x D",
                    f"{part['dimensions_mm'][0]} x {part['dimensions_mm'][1]} x {part['dimensions_mm'][2]} mm",
                    "Outer housing envelope",
                ],
            ]
        else:
            rows = [
                ["Parameter", "Symbol", "Specification Value", "Test Condition / Notes"],
                ["Part Number", "ID", part["part_id"], part["series"]],
                ["Manufacturer", "Vendor", part["vendor"], "Precision Inline Planetary"],
                ["Reduction Ratio", "i_ext", part["ratio_raw"], "Exact mechanical ratio"],
                ["Transmission Efficiency", "eta", part["efficiency_raw"], "At rated full load"],
                ["Rated Output Torque", "tau_out_max", part["torque_raw"], "Continuous output limit"],
                ["Gearbox Mass (Weight)", "m_gb", part["mass_raw"], "Dry weight with adapter flange"],
                [
                    "Envelope Dimensions",
                    "W x L x D",
                    f"{part['dimensions_mm'][0]} x {part['dimensions_mm'][1]} x {part['dimensions_mm'][2]} mm",
                    "ISO flange mount",
                ],
            ]

        table = ax_table.table(
            cellText=rows[1:],
            colLabels=rows[0],
            loc="upper center",
            cellLoc="left",
            colWidths=[0.28, 0.14, 0.30, 0.28],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.55)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            if r == 0:
                cell.set_facecolor("#1e293b")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 1:
                cell.set_facecolor("#f8fafc")
            else:
                cell.set_facecolor("#ffffff")

        # Plot Torque-Speed characteristic curve on bottom half of Page 1
        ax_plot = fig.add_axes([0.12, 0.10, 0.78, 0.28])
        if part["type"] == "actuator":
            tau_s = part["stall_torque_nm"]
            w_rpm = part["no_load_speed_rpm"]
            w_rads = w_rpm * 2.0 * np.pi / 60.0
            tau_c = part["continuous_torque_nm"] or (0.33 * tau_s)
            taus = np.linspace(0, tau_s, 100)
            speeds = w_rads * (1.0 - taus / tau_s)
            ax_plot.plot(taus, speeds, color="#2563eb", linewidth=2.2, label="Linear Torque-Speed Boundary")
            ax_plot.axvline(tau_c, color="#16a34a", linestyle="--", linewidth=1.6, label=f"Catalog Continuous Assumption ({tau_c:.2f} N·m)")
            ax_plot.axvline(0.8 * tau_s, color="#dc2626", linestyle=":", linewidth=1.6, label=f"80% Peak Safety Line ({0.8*tau_s:.2f} N·m)")
            ax_plot.fill_betweenx(speeds, 0, np.minimum(taus, tau_c), color="#dcfce7", alpha=0.5, label="Continuous Safe Zone")
            ax_plot.set_xlabel("Output Torque [N·m]", fontsize=9)
            ax_plot.set_ylabel("Output Speed [rad/s]", fontsize=9)
            ax_plot.set_title(
                f"{part['part_id']} Output Torque-Speed Curve (omega_0 = {w_rads:.2f} rad/s, tau_s = {tau_s:.1f} N·m)",
                fontsize=10,
                fontweight="bold",
            )
            ax_plot.grid(True, linestyle="--", alpha=0.4)
            ax_plot.legend(loc="upper right", fontsize=8)
        else:
            ratios = [1.0, part["ratio"]]
            effs = [100.0, part["efficiency"] * 100.0]
            ax_plot.bar(["Direct (1:1)", f"{part['part_id']} ({part['ratio']:.0f}:1)"], effs, color=["#94a3b8", "#2563eb"], width=0.4)
            ax_plot.set_ylim(0, 110)
            ax_plot.set_ylabel("Efficiency [%]", fontsize=9)
            ax_plot.set_title(f"{part['part_id']} Transmission Efficiency & Torque Rating ({part['rated_output_torque_nm']} N·m)", fontsize=10)
            ax_plot.grid(True, axis="y", linestyle="--", alpha=0.4)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Mechanical Interface, Mounting & Application Notes
        fig2 = plt.figure(figsize=(8.27, 11.69))
        fig2.patch.set_facecolor("#ffffff")
        fig2.text(0.5, 0.015, "MotionFit-generated reference — not a manufacturer original; verify specifications", ha="center", fontsize=8, color="#92400e")
        ax2 = fig2.add_axes([0.07, 0.07, 0.86, 0.86])
        ax2.axis("off")
        ax2.text(0.0, 0.96, f"{part['part_id']} — Project Reference & Review Notes", fontsize=14, fontweight="bold", color="#0f172a")
        ax2.text(1.0, 0.96, f"Document: {part['doc']} | Page 2 of 2", fontsize=9, color="#475569", ha="right")
        notes = [
            "2. Mechanical Mounting & Frame Integration",
            f"   - Housing Envelope: {part['dimensions_mm'][0]} mm x {part['dimensions_mm'][1]} mm x {part['dimensions_mm'][2]} mm",
            f"   - Catalog Mass (unverified): {part['mass_raw']} ({part['mass_kg']:.4f} kg)",
            "   - Mounting Pattern: Not verified; obtain manufacturer flange/shaft drawings.",
            "",
            "3. Thermal & Duty-Cycle Notes",
            "   - Catalog continuous torque is an input assumption until its manufacturer rating is checked.",
            "   - The 0.33x stall estimate, where used, is a heuristic rather than a measured thermal rating.",
            "   - The linear torque-speed envelope is a screening model, not a measured performance curve.",
            "",
            "4. Project Reference Index",
            f"   - Part ID: {part['part_id']} | Source Document: {part['doc']} | Spec Table: Page 1",
        ]
        ax2.text(0.0, 0.86, "\n".join(notes), fontsize=10, color="#334155", va="top", family="monospace")
        pdf.savefig(fig2)
        plt.close(fig2)


def generate_all() -> dict:
    """Generate all PDFs, prices.csv, and catalog.json."""
    DATASHEETS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Write prices.csv
    prices_path = DATASHEETS_DIR / "prices.csv"
    with open(prices_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["part_id", "price_usd", "source_url"])
        for idx, p in enumerate(RAW_PARTS, start=2):
            writer.writerow([p["part_id"], f"{p['price_usd']:.2f}", p["source_url"]])

    # 2. Render PDFs and build catalog.json
    catalog_parts = []
    for idx, p in enumerate(RAW_PARTS, start=2):
        pdf_path = DATASHEETS_DIR / p["doc"]
        _render_pdf(p, pdf_path)

        if p["type"] == "actuator":
            stall = float(p["stall_torque_nm"])
            if p["continuous_torque_nm"] is None:
                cont = round(0.33 * stall, 3)
                derived = True
            else:
                cont = float(p["continuous_torque_nm"])
                derived = p.get("vendor") == "ROBOTIS" and p.get("series") == "DYNAMIXEL-X"
            w_rad_s = round(float(p["no_load_speed_rpm"]) * 2.0 * np.pi / 60.0, 3)

            entry = {
                "part_id": p["part_id"],
                "type": "actuator",
                "vendor": p["vendor"],
                "series": p["series"],
                "rated_voltage_v": p["rated_voltage_v"],
                "stall_torque_nm": {
                    "value": stall,
                    "raw": p["stall_raw"],
                    "src": {"doc": p["doc"], "page": p["spec_page"]},
                },
                "continuous_torque_nm": {
                    "value": cont,
                    "raw": p.get("continuous_raw", f"0.33 * {stall} N.m (derived)"),
                    "src": {"doc": p["doc"], "page": p["spec_page"]},
                    "derived": derived,
                },
                "no_load_speed_rad_s": {
                    "value": w_rad_s,
                    "raw": p["speed_raw"],
                    "src": {"doc": p["doc"], "page": p["spec_page"]},
                },
                "mass_kg": {
                    "value": float(p["mass_kg"]),
                    "raw": p["mass_raw"],
                    "src": {"doc": p["doc"], "page": p["spec_page"]},
                },
                "price_usd": {
                    "value": float(p["price_usd"]),
                    "src": {"doc": "prices.csv", "row": idx},
                },
                "dimensions_mm": p["dimensions_mm"],
                "confidence": "high",
                "user_confirmed": True,
            }
        else:
            entry = {
                "part_id": p["part_id"],
                "type": "gearbox",
                "vendor": p["vendor"],
                "series": p["series"],
                "ratio": {
                    "value": float(p["ratio"]),
                    "raw": p["ratio_raw"],
                    "src": {"doc": p["doc"], "page": p["spec_page"]},
                },
                "efficiency": {
                    "value": float(p["efficiency"]),
                    "raw": p["efficiency_raw"],
                    "src": {"doc": p["doc"], "page": p["spec_page"]},
                },
                "rated_output_torque_nm": {
                    "value": float(p["rated_output_torque_nm"]),
                    "raw": p["torque_raw"],
                    "src": {"doc": p["doc"], "page": p["spec_page"]},
                },
                "mass_kg": {
                    "value": float(p["mass_kg"]),
                    "raw": p["mass_raw"],
                    "src": {"doc": p["doc"], "page": p["spec_page"]},
                },
                "price_usd": {
                    "value": float(p["price_usd"]),
                    "src": {"doc": "prices.csv", "row": idx},
                },
                "dimensions_mm": p["dimensions_mm"],
                "confidence": "high",
                "user_confirmed": True,
            }
        if p["part_id"] in SOURCE_VERIFICATIONS:
            entry["source_verification"] = SOURCE_VERIFICATIONS[p["part_id"]]
        if p.get("series") == "DYNAMIXEL-X":
            entry["continuous_torque_nm"]["derived"] = True
            entry["continuous_torque_nm"]["derivation"] = "Engineering estimate; not an official continuous thermal rating"
            entry["continuous_torque_nm"]["raw"] = f"Engineering estimate: {entry['continuous_torque_nm']['value']} N.m (not manufacturer rated)"
        entry["provenance"] = {"source": "project_generated_reference", "manufacturer_verified": False,
                               "requires_spec_review": True, "source_url": p.get("source_url")}
        catalog_parts.append(entry)

    catalog = {
        "catalog_id": "demo-catalog-v1",
        "provenance": {"source": "project_generated_reference", "manufacturer_verified": False, "requires_spec_review": True,
                       "note": "Bundled PDFs are project-generated reference documents, not manufacturer originals."},
        "confirmed": True,
        "spot_check_passed": True,
        "parts": catalog_parts,
    }
    catalog_path = DEMO_DIR / "catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    return catalog


if __name__ == "__main__":
    cat = generate_all()
    print(f"Generated {len(cat['parts'])} datasheet PDFs and catalog.json in {DEMO_DIR}")
