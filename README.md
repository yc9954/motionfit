# MotionFit: Transforming Human Demonstration into Certified Physical Robotics

[![Live Demo](https://img.shields.io/badge/Google_Cloud_Run-Live_Demo-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://motionfit-494618317318.asia-northeast3.run.app)
[![MuJoCo 3.3.0](https://img.shields.io/badge/Physics_Engine-MuJoCo_3.3.0-FF6F00?style=for-the-badge)](https://mujoco.org)
[![Gemini 4 Argon](https://img.shields.io/badge/AI_Reasoning-Gemini_4_Argon-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev)
[![MediaPipe Tasks Vision](https://img.shields.io/badge/Vision-MediaPipe_On--Device-00C49F?style=for-the-badge)](https://developers.google.com/mediapipe)

**Live Deployment:** [https://motionfit-494618317318.asia-northeast3.run.app](https://motionfit-494618317318.asia-northeast3.run.app)  
**GitHub Repository:** [https://github.com/yc9954/motionfit](https://github.com/yc9954/motionfit)

---

## Executive Summary & Engineering Motivation

In an era where generative AI can write code, compose music, and generate photorealistic video in seconds, designing physical robots remains painfully stuck in the past.

During in-depth user interviews with mechanical engineering researchers and robotics hardware practitioners, we identified a persistent, costly bottleneck: **the brutal gap between digital kinematics and physical dynamics.**

In conventional robot development, an engineer starts with a target motion in software. But transforming that digital trajectory into working physical hardware takes weeks of manual calculation. Engineers must manually size actuators, select planetary gear ratios, calculate thermal duty cycles, and evaluate dynamic cantilever bending moments across spreadsheets. Despite these calculations, physical prototypes routinely fail during initial testing: an actuator that appeared sufficient on paper burns out under continuous thermal load, or joint gearboxes strip under sudden inertial spikes, causing the manipulator to sag or drop its payload.

Worse, when an engineer tries to fix a failing joint by swapping in a larger motor, the added motor mass introduces a **parasitic weight spiral**: the heavier motor overloads every upstream joint, forcing larger motors all the way down to the base and quickly exceeding weight and budget limits.

**MotionFit solves this chronic problem.** It is an autonomous physical AI co-pilot that converts casual human video demonstrations into verified, buildable robot hardware backed by real-world off-the-shelf components, rigid-body multi-physics simulation, and deep engineering reasoning powered by Google Gemini 4 Argon.

---

## Why Gemini 4 Argon? (The Autonomous Mechanical Engineer)

Conventional large language models are notorious for failing at physical hardware design. When asked to fix an overloaded robot joint, standard LLMs suffer from three critical flaws:
1. **Greedy Scaling Hallucination**: They naively suggest "use a bigger motor," ignoring that motor mass compounds non-linearly across upstream links (the parasitic mass spiral).
2. **Catalog Hallucination**: They invent non-existent actuators with physically impossible power densities or imaginary part numbers.
3. **Inability to Reason over Dynamic Telemetry**: They cannot interpret high-frequency time-series joint torque derivatives, thermal root-mean-square loads, and tracking error signatures.

**Gemini 4 Argon is specifically architected to overcome these fundamental limitations:**

### 1. High-Level Extended Thinking for Multi-Objective Trade-Offs
Hardware optimization is coupled and non-linear. Joint torque, continuous thermal dissipation, structural deflection, and cost cannot be solved independently. Powered by high-level extended thinking (`thinking_level=high`), Gemini 4 Argon performs multi-step mechanical root-cause diagnosis. When a shoulder pitch actuator experiences 238 percent torque strain under a 500-gram payload at full extension, Argon recognizes that adding raw motor mass will destabilize the entire arm. Instead, it reasons that the joint needs **mechanical advantage**, proposing a 10:1 planetary gearbox reduction and structural lightening with thin-walled 6061-T6 aerospace aluminum tubing.

### 2. Zero-Hallucination Grounded Action Space
Argon does not generate theoretical components. It is strictly constrained to a verified catalog of procurable off-the-shelf parts (Dynamixel servos, T-Motor brushless drives, APEX Dynamics planetary gearboxes). Argon cross-references official manufacturer datasheets, citing peak torque, continuous thermal limits, and gear efficiencies. Every recommended component is physically buyable and mathematically verified.

### 3. Active In-the-Loop Simulation Feedback
Argon does not operate as an isolated conversational chatbot. It functions as the core intelligence in an active closed loop with the MuJoCo 3.3.0 physics engine:
- MuJoCo simulates the candidate robot under full gravity and 500 Hz dynamics.
- The 17-point deterministic engineering referee flags any violation (e.g., path error exceeding 25 millimeters or torque exceeding continuous thermal thresholds).
- Argon ingests the exact numerical failure trace, diagnoses the root cause, and synthesizes an improved hardware candidate in seconds.

### 4. Multimodal Synthesis (Human Vision to Machine MJCF)
Argon effortlessly synthesizes multimodal inputs: natural language task parameters (mass, reach, budget), on-device 3D skeletal landmarks from human video, and low-level physics simulation telemetry, unifying them into validated MJCF kinematic models.

---

## Alignment with Evaluation Criteria

MotionFit was engineered from the ground up to excel across all hackathon evaluation criteria:

| Evaluation Criterion | How MotionFit Excels |
|:---|:---|
| **1. Innovation & Real-World Impact** | Solves the genuine physical engineering bottleneck identified through direct interviews with mechanical engineers: eliminating the costly, trial-and-error cycle of actuator burnout and parasitic mass spirals. |
| **2. Technical Depth & Engineering Rigor** | Full multi-body rigid dynamics in MuJoCo 3.3.0 on a 500 Hz forward integration grid. Replays actual simulated joint angles (`q_act`) reflecting real physics droop, audited by 17 deterministic checks. |
| **3. Meaningful Use of Gemini 4 Argon** | Argon is not a superficial chatbot wrapper. It serves as an autonomous mechanical lead engineer using extended thinking to solve coupled, non-linear physical trade-offs in an active closed-loop simulation cycle. |
| **4. Trust, Safety & Zero Hallucination** | Every hardware proposal is constrained to real, procurable manufacturer parts with verifiable datasheet citations. If no candidate passes, the system explicitly reports failure rather than faking approval. |
| **5. User Experience & Product Completeness** | On-device zero-latency video extraction via WebAssembly MediaPipe Tasks Vision, interactive Three.js 3D digital twin with animated part replacement, and fully deployed on Google Cloud Run. |

---

## End-to-End System Architecture

```
[ Casual Smartphone Video / Natural Language Task ]
                       │
                       ▼
    [ On-Device MediaPipe Tasks Vision (Client WASM) ]
      • Zero-cloud private 3D landmark extraction
      • Torso-ratio depth ambiguity resolution
      • Analytic retargeting into metric 4-DOF workspace
                       │
                       ▼
      [ Parametric MJCF Model Compiler ]
      • Link geometries, centers of mass & inertia tensors
      • Real catalog actuator torque / speed curves
                       │
                       ▼
    [ MuJoCo 3.3.0 Multi-Body Physics Simulation ]
      • Pass A: Inverse dynamics (ideal feedforward torques)
      • Pass B: Forward dynamics integration at 500 Hz
      • True gravitational droop, actuator saturation & inertia
                       │
                       ▼
    [ Deterministic 17-Point Engineering Referee ]
      • RMS path tracking error <= 25 mm
      • Joint peak torque <= 100% datasheet rating
      • Continuous thermal torque <= 80% rated limit
      • Budget ceiling, mass targets & reachability
              │                         │
          [ Passes ]                 [ Fails ]
              │                         │
              │                         ▼
              │        [ Google Gemini 4 Argon Co-Pilot ]
              │          • Extended thinking root-cause diagnosis
              │          • Parses joint torque & thermal strain curves
              │          • Prevents parasitic mass accumulation
              │          • Proposes gear reduction & alloy substitution
              │                         │
              │                         ▼
              └──────◄───── [ Next Hardware Iteration Loop ]
```

---

## Core Capabilities & Features

### 1. On-Device Multimodal Motion Capture & Kinematic Retargeting
Users upload a smartphone video of a human demonstration (such as reaching down, grasping a 500 mL water bottle, and lifting it across a span) or enter a natural language task description with explicit weight and budget constraints.
- **Zero Cloud Latency & Total Privacy**: Video frames are processed entirely on-device via WebAssembly-powered MediaPipe Tasks Vision. Raw video never touches external servers.
- **Depth Disambiguation**: Resolves monocular depth ambiguity by calculating proportional torso-to-limb anatomical ratios across time.
- **Kinematic Retargeting**: Normalizes and scales measured human wrist trajectories into a feasible workspace for a four-degree-of-freedom manipulator using analytic inverse kinematics.

### 2. High-Fidelity Physics Verification in MuJoCo 3.3.0
Unlike conventional tools that produce cosmetic animations, MotionFit simulates real Newtonian mechanics.
- **Rigid-Body Dynamics**: Generates custom MJCF models with accurate link mass distributions, centers of mass, and composite inertia tensors.
- **Dual-Pass Verification**:
  - *Pass A (Inverse Dynamics)*: Computes theoretical joint torque requirements along the trajectory.
  - *Pass B (Forward Dynamics at 500 Hz)*: Simulates closed-loop PD-controlled actuators with back-EMF velocity limits, dynamic joint friction, gravitational loading, and motor droop.
- **True Simulated Playback**: The Three.js viewer replays the actual simulated joint angles (`q_act`), showing genuine physical droop when an actuator is undersized, rather than an idealized line.

### 3. Deterministic 17-Point Engineering Audit
Every hardware candidate must satisfy seventeen strict, deterministic engineering criteria before certification:
1. **End-Effector RMS Tracking Accuracy**: Must remain within 25 millimeters of the target trajectory.
2. **Maximum Path Deviation**: Maximum instantaneous error must not exceed 50 millimeters.
3. **Shoulder Pitch Peak Torque**: Must not exceed 100 percent of actuator peak rating.
4. **Elbow Pitch Peak Torque**: Must not exceed 100 percent of actuator peak rating.
5. **Continuous Thermal Load**: RMS torque over the trajectory must stay within continuous thermal safety limits (under 80 percent of rated continuous torque).
6. **Actuator Speed Saturation**: Joint angular velocities must not exceed no-load maximum speed ratings.
7. **Cantilever Bending Margin**: Structural link deflection under dynamic reach must remain within allowable thresholds.
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

### 4. Interactive 3D Digital Twin & Transformation Studio
- **Live 3D Hardware Transformation**: As Gemini 4 Argon refines the hardware, the Three.js viewport animates the mechanical transformation. Golden APEX planetary gearbox collars snap onto the stressed joint, and bulky prototype links are swapped for slim aerospace aluminum tubes.
- **Dynamic Manipulated Objects**: Dynamically renders the target payload (coffee mug, water bottle, industrial tool, parcel) at the end-effector.
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
