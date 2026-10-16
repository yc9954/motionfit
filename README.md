# MotionFit: Transforming Human Demonstration into Certified Physical Robotics

[![Live Demo](https://img.shields.io/badge/Google_Cloud_Run-Live_Demo-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://motionfit-494618317318.asia-northeast3.run.app)
[![MuJoCo 3.3.0](https://img.shields.io/badge/Physics_Engine-MuJoCo_3.3.0-FF6F00?style=for-the-badge)](https://mujoco.org)
[![Gemini 4 Argon](https://img.shields.io/badge/AI_Reasoning-Gemini_4_Argon-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev)
[![MediaPipe Tasks Vision](https://img.shields.io/badge/Vision-MediaPipe_On--Device-00C49F?style=for-the-badge)](https://developers.google.com/mediapipe)

**Live Deployment:** [https://motionfit-494618317318.asia-northeast3.run.app](https://motionfit-494618317318.asia-northeast3.run.app)  
**GitHub Repository:** [https://github.com/yc9954/motionfit](https://github.com/yc9954/motionfit)

---

## Overview

In an era where generative AI can write code, compose music, and generate photorealistic video in seconds, designing physical robots remains painfully stuck in the past. 

During in-depth user interviews with mechanical engineering researchers and robotics hardware practitioners, we identified a persistent, costly bottleneck: **the brutal gap between digital kinematics and physical dynamics.** 

In conventional robot design, an engineer starts with a target motion in software, but transforming that trajectory into real hardware takes weeks of manual calculation. Engineers must manually size motors, select gear ratios, evaluate thermal duty cycles, and calculate dynamic cantilever bending moments across spreadsheets. Despite these calculations, physical prototypes routinely fail during initial testing: an actuator that appeared sufficient on paper burns out under continuous thermal load, or joint gearboxes strip under sudden inertial spikes, causing the manipulator to sag or drop its payload.

**MotionFit eliminates this bottleneck.** It is an autonomous physical AI co-pilot that converts casual human video demonstrations into verified, buildable robot hardware backed by real-world off-the-shelf components and rigid-body multi-physics simulation.

---

## Key Highlights

- **User-Centric Engineering Motivation**: Born from direct interviews with mechanical engineers who lose weeks sizing actuators and dealing with prototype burnout caused by parasitic mass spirals.
- **On-Device Multimodal Understanding**: MediaPipe Tasks Vision extracts 3D skeletal landmarks locally in the browser with zero cloud latency and complete user privacy.
- **Closed-Loop Rigid-Body Dynamics in MuJoCo 3.3.0**: Simulates the synthesized manipulator under gravity, dynamic inertia, and electro-mechanical actuator droop curves on a high-frequency 500 Hz simulation grid.
- **Deterministic 17-Point Engineering Referee**: Audits root-mean-square tracking accuracy, peak torque margins, continuous thermal thresholds, velocity saturation, and budget constraints before any design is certified.
- **Deep Root-Cause Hardware Reasoning with Gemini 4 Argon**: Uses high-level extended thinking to diagnose failures and navigate verified hardware catalogs (Dynamixel, T-Motor, APEX Dynamics), preventing destructive mass accumulation.
- **Interactive 3D Digital Twin**: Live Three.js visualization that physically swaps prototype links, snaps golden planetary gearbox collars into place, and displays real physics droop versus certified tracking.

---

## End-to-End System Architecture

```
[ Human Video / Natural Language Task ]
                  │
                  ▼
[ On-Device MediaPipe Vision (Browser WASM) ]
  • 3D body landmark extraction
  • Anatomical torso ratio depth resolution
  • Retargeting into metric 4-DOF robot workspace
                  │
                  ▼
[ Parametric MJCF Model Compilation ]
  • Link geometry & composite inertia tensors
  • Real catalog actuator torque / speed curves
                  │
                  ▼
[ MuJoCo 3.3.0 Multi-Body Physics Simulation ]
  • Inverse dynamics (ideal feedforward torque)
  • Forward dynamics integration at 500 Hz
  • True gravity droop, torque saturation & backlash
                  │
                  ▼
[ Deterministic 17-Point Engineering Referee ]
  • RMS path error <= 25 mm
  • Joint peak torque <= 100% datasheet limit
  • Continuous thermal torque <= 80% rating
  • Budget & payload constraints
           │                      │
       [ Passes ]              [ Fails ]
           │                      │
           │                      ▼
           │     [ Google Gemini 4 Argon Co-Pilot ]
           │       • Extended thinking root-cause diagnosis
           │       • Analyzes joint torque / thermal strain
           │       • Avoids parasitic mass spirals
           │       • Proposes gearbox reduction / lightweight alloy
           │                      │
           │                      ▼
           └───◄── [ Next Hardware Iteration Loop ]
```

---

## Core Capabilities

### 1. On-Device Multimodal Motion Capture & Kinematic Retargeting
Users can upload a casual smartphone video of an arm performing a manipulation task (such as reaching down, grasping a water bottle, and lifting it across a span) or provide a natural language specification with explicit mass and budget requirements.
- **Zero Cloud Latency & Total Privacy**: Video frames are processed entirely on-device via WebAssembly-powered MediaPipe Tasks Vision. Video bytes never leave the client.
- **Depth Disambiguation**: Solves monocular depth ambiguity by calculating proportional torso-to-limb ratios over time.
- **Kinematic Retargeting**: Normalizes and scales measured human wrist trajectories into a feasible workspace for a four-degree-of-freedom manipulator using analytic inverse kinematics.

### 2. High-Fidelity Physics Verification in MuJoCo 3.3.0
Unlike conventional generative AI tools that hallucinate hardware specifications, MotionFit grounds every decision in Newtonian physics.
- **Rigid-Body Dynamics**: Constructs a custom MJCF definition with exact link mass distributions, center-of-mass offsets, and inertia tensors.
- **Dual-Pass Evaluation**:
  - *Pass A (Inverse Dynamics)*: Computes the theoretical torque profiles required along the target trajectory.
  - *Pass B (Forward Dynamics at 500 Hz)*: Simulates a closed-loop PD controller with actuator voltage limits, dynamic friction, gravitational loading, and motor droop.
- **Real Playback**: The 3D viewer replays the actual simulated joint angles (`q_act`), showing genuine physical droop when an actuator is undersized, rather than an idealized trajectory.

### 3. Deterministic 17-Point Engineering Audit
Every hardware candidate must pass seventeen strict, deterministic engineering criteria before receiving approval:
1. **End-Effector RMS Tracking Accuracy**: Must remain within 25 millimeters of the demonstration trajectory.
2. **Maximum Path Deviation**: Maximum instantaneous error must not exceed 50 millimeters.
3. **Shoulder Pitch Peak Torque**: Must not exceed 100 percent of actuator peak rating.
4. **Elbow Pitch Peak Torque**: Must not exceed 100 percent of actuator peak rating.
5. **Continuous Thermal Load**: RMS torque over the trajectory must stay within continuous thermal safety limits (under 80 percent of rated continuous torque).
6. **Actuator Speed Saturation**: Joint angular velocities must not exceed no-load maximum speed ratings.
7. **Cantilever Bending Margin**: Structural link deflection under full dynamic reach must remain within allowable thresholds.
8. **Payload Holding Margin**: Static holding torque at maximum extension must support the target payload plus a 20 percent safety margin.
9. **Gravitational Sag Tolerance**: Terminal position error under static payload hold must be under 10 millimeters.
10. **Actuator Temperature Headroom**: Estimated coil temperature rise under continuous duty cycle must remain within manufacturer limits.
11. **Total Arm Structural Mass**: Must not exceed the specified maximum system mass budget.
12. **Workspace Reachability**: Manipulator workspace envelope must fully encompass all task waypoints.
13. **Joint Limit Compliance**: Angular excursions must stay strictly within mechanical joint limits.
14. **Kinematic Singularity Margin**: Manipulator Jacobian condition number must remain safely away from singular configurations.
15. **Total Hardware Bill of Materials Cost**: Total cost must remain strictly within the user's defined financial budget.
16. **Datasheet Citation Integrity**: All component properties must map to verified manufacturer datasheets.
17. **Zero Hallucinated Components**: Component selections are strictly constrained to real, procurable off-the-shelf parts.

### 4. Deep Engineering Reasoning with Gemini 4 Argon
Hardware optimization is non-linear and coupled. Upgrading to a heavier motor increases torque, but the added parasitic mass overloads all upstream joints, causing a cascading design failure.
- **Extended Thinking**: With `thinking_level=high`, Gemini 4 Argon conducts root-cause analysis across time-series torque telemetry.
- **Mechanical Advantage over Raw Mass**: When a shoulder pitch joint experiences 238 percent torque overload, Argon recognizes that adding raw motor mass is counterproductive. Instead, it introduces a 10:1 planetary reduction gearbox and recommends switching the link structure to thin-walled 6061-T6 aerospace aluminum tubing.
- **Strictly Grounded Catalog**: Argon selects exclusively from verified manufacturer parts:
  - **Dynamixel Actuators**: XL430-W250, XM430-W350, XH540-W270
  - **T-Motor Brushless Actuators**: AK80-9, AK70-10
  - **APEX Dynamics Planetary Gearboxes**: High-precision low-backlash reducers
  - **Structural Materials**: 6061-T6 Aluminum, Carbon Fiber Tubing

### 5. Interactive 3D Digital Twin & Transformation Studio
- **Live 3D Hardware Replacement**: As Gemini 4 Argon refines the hardware, the Three.js viewport animates the mechanical transformation. Golden APEX planetary gearbox collars snap onto the stressed joint, and bulky prototype links are swapped for slim aerospace aluminum tubes.
- **Dynamic Manipulated Objects**: Renders the exact payload (coffee mug, water bottle, industrial tool, parcel) at the end-effector.
- **Interactive Assistant Mascot**: An animated SVG mesh gradient mascot tracks user interaction and signals simulation status.

---

## Verified Off-the-Shelf Hardware Catalog

| Component | Category | Mass | Peak Torque | Continuous Torque | Max Speed | Cost |
|:---|:---|:---|:---|:---|:---|:---|
| **Dynamixel XL430-W250** | Smart Actuator | 57 g | 1.5 N·m | 0.8 N·m | 61 rpm | $49.90 |
| **Dynamixel XM430-W350** | High-Res Actuator | 82 g | 4.1 N·m | 1.8 N·m | 46 rpm | $249.00 |
| **Dynamixel XH540-W270** | Heavy-Duty Servo | 165 g | 10.6 N·m | 3.5 N·m | 30 rpm | $429.00 |
| **T-Motor AK80-9** | Brushless Actuator | 485 g | 9.0 N·m | 3.2 N·m | 240 rpm | $380.00 |
| **APEX Dynamics 10:1** | Planetary Gearbox | 120 g | 35.0 N·m rating | 18.0 N·m rating | 97% efficiency | $165.00 |
| **6061-T6 Aluminum** | Structural Tubing | 0.28 kg/m | High yield | High stiffness | - | $18.00/m |

---

## Project Structure

```
motionfit/
├── backend/
│   ├── main.py              # FastAPI server, SSE event streaming, lifecycle management
│   ├── sim.py               # MuJoCo 3.3.0 physics simulation & numerical integration
│   ├── checker.py           # 17 deterministic engineering acceptance criteria
│   ├── gemini.py            # Gemini 4 Argon client with high-level extended thinking
│   ├── loop.py              # Iterative diagnose-propose-simulate verification loop
│   ├── model_builder.py     # Procedural MJCF model compiler & composite inertia tensors
│   ├── motion.py            # Kinematic retargeting & trajectory spline interpolation
│   ├── kinematics.py        # Analytic 4-DOF forward and inverse kinematics
│   ├── task_planner.py      # Natural language task specification parser
│   ├── validate.py          # Datasheet cross-referencing & citation enforcement
│   └── storage.py           # Atomic project persistence & versioned artifact locking
├── frontend/
│   ├── src/
│   │   ├── main.ts          # Studio application controller & state management
│   │   ├── viewer.ts        # Three.js 3D digital twin, MJCF parsing, hardware animations
│   │   ├── pose.ts          # MediaPipe Tasks Vision on-device pose extraction
│   │   ├── workspace.ts     # Project workspace, telemetry charts & candidate comparison
│   │   └── components/ui/   # Mesh gradient SVG mascot & interactive widgets
│   ├── index.html           # Single-page studio interface
│   ├── package.json         # Frontend dependencies (Three.js, MediaPipe, Lucide)
│   └── vite.config.ts       # Vite configuration with API proxying
├── catalog/                 # Verified manufacturer component datasheets & pricing
├── tests/                   # 44 automated backend verification tests
├── Dockerfile               # Multi-stage production container build
├── requirements.txt         # Pinned backend dependencies
└── README.md
```

---

## Quick Start & Local Setup

### Prerequisites
- Python 3.11 or 3.12
- Node.js 18+ and npm
- Valid Google Gemini API Key (optional for offline mode)

### 1. Clone and Install Backend

```bash
git clone https://github.com/yc9954/motionfit.git
cd motionfit

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install and Build Frontend

```bash
cd frontend
npm ci
npm run build
cd ..
```

### 3. Run the Application

```bash
# Start backend server on port 8080
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080` in your browser.

To run with full Gemini 4 Argon reasoning enabled:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export MOTIONFIT_ARGON_MODEL="gemini-2.5-flash" # or your preferred Gemini model
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080
```

### 4. Offline / Deterministic Mode
To test the physics engine and deterministic referee without an external API key:
```bash
MOTIONFIT_OFFLINE=1 .venv/bin/python -m pytest -q
```
All 44 automated tests will run and pass deterministically.

---

## Production Cloud Run Deployment

MotionFit is containerized via a multi-stage Dockerfile and deployed on Google Cloud Run in the `asia-northeast3` (Seoul) region:

```bash
gcloud run deploy motionfit \
  --source . \
  --region asia-northeast3 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 900 \
  --concurrency 4 \
  --min-instances 1 \
  --max-instances 1 \
  --no-cpu-throttling \
  --allow-unauthenticated \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

Live service: [https://motionfit-494618317318.asia-northeast3.run.app](https://motionfit-494618317318.asia-northeast3.run.app)

---

## Engineering Impact

MotionFit proves that generative physical AI should not be about generating fictional 3D meshes or guessing actuator capabilities. By anchoring high-level extended thinking in rigid-body multi-physics simulation and verified off-the-shelf component catalogs, MotionFit bridges human demonstration to certified physical robotics in minutes.
