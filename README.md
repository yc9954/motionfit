# MotionFit

> **Show it the motion. MotionFit tells you which robot-arm parts to change, and proves it in simulation.**

MotionFit is an end-to-end hardware verification and actuator-selection loop powered by **Gemini 4 Argon** (`gemini-4-argon` via the `google-genai` Interactions API) and **MuJoCo** physics simulation.

## Problem

Choosing actuators for a serial robot arm is a tightly coupled physics problem: upgrading a failing elbow motor adds distal mass, which increases the gravity and inertial torque required at the shoulder, potentially pushing the shoulder past its thermal or stall limit. Teams typically discover these coupling failures only after ordering hardware.

## How MotionFit Works

1. **Inputs**: Upload an initial arm specification (`design.json`), a folder of actuator & gearbox datasheet PDFs + `prices.csv`, and a 5–15 s video of a human performing the desired motion.
2. **Human-in-the-Loop Review**:
   - **Catalog Extractor** (`gemini-4-argon`, `thinking_level="low"`, structured output) extracts stall torque, continuous torque, no-load speed, mass, and page citations (`{doc, page}`) from every PDF into `catalog.json`. An engineer reviews and confirms the cited table.
   - **Motion & Intent Analysis**: MediaPipe extracts 3D arm landmarks while `gemini-4-argon` (`thinking_level="medium"`) segments the video into task phases and critical high-load/high-speed windows.
3. **Physics Referee & Argon Design Loop**:
   - **Deterministic Referee (`sim.py` + `checker.py`)**: Builds a parametric 4-DOF MuJoCo model (`arm4.xml.j2`), runs analytic IK, Pass A inverse dynamics (`mj_inverse`), and Pass B 500 Hz torque-speed-limited forward simulation (`mj_step`).
   - **Coupled Hardware Reasoning (`loop.py` + `gemini.py`)**: When a design fails, `gemini-4-argon` (`thinking_level="high"`, stateful via `previous_interaction_id`) diagnoses the root cause and proposes at most 3 cited catalog swaps or link adjustments.
   - **Proposal Validator (`validate.py`)**: Deterministically verifies every part ID, parameter bound, budget limit, and cited datasheet value (within 2%) before the next simulation iteration.
4. **Synchronized 3-Pane CAD Compare & Engineering Report**:
   - Compare the source video, the drooping original arm (red saturated joints), and the verified redesigned arm (green joints tracking the ghost trajectory within 15 mm RMSE) in a CAD-grade Three.js viewer.
   - Generate a fully cited HTML engineering report (`gemini-3.8-flash` for connecting prose, zero invented numbers), record engineer sign-off, and export the verified bundle (`design_final.json`, `arm_final.xml`, `metrics.json`, `report.html`).

## Quick Start (Local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the physics & loop acceptance tests
pytest -v

# Start the server
uvicorn backend.main:app --host 0.0.0.0 --port 8080
```

## Cloud Run Deployment

```bash
gcloud run deploy motionfit --source . --region asia-northeast3 \
  --memory 4Gi --cpu 2 --timeout 900 --concurrency 4 --min-instances 1 \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
  --allow-unauthenticated
```
