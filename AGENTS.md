# AGENTS.md — MotionFit Build Log, Alignment & Technical Lessons

## M0 Alignment Decisions (Recorded 2026-10-16)

1. **Dataset & Catalog Generation**:
   - Real specifications sourced from ROBOTIS DYNAMIXEL X-series (`XL430-W250`, `XM430-W210`, `XM430-W350`, `XH430-W210`, `XH430-W350`, `XM540-W150`, `XM540-W270`, `XH540-W150`, `XH540-W270`), DYNAMIXEL-P series (`PH42-020-S300-R`, `PH54-100-S500-R`), CubeMars AK quasi-direct-drive actuators (`AK60-6`, `AK70-10`, `AK80-9`), and inline planetary gearboxes (`APEX-PG20-2`, `APEX-PG20-3`).
   - Multi-page and single-page PDF datasheets (`backend/data/datasheets/*.pdf`) and `prices.csv` are generated with authentic numbers and page citations so both live PDF citation links (`#page=N`) and live Argon catalog extraction work out of the box.
2. **Repository & Deployment**:
   - Local repository rooted at `/Users/yuchanlee/motionfit`.
   - Single-container Cloud Run architecture (`Dockerfile` building Vite frontend into `static/` served by FastAPI `backend/main.py`).
3. **Gemini API & Models (`google-genai` Interactions API)**:
   - Core design loop (`high` thinking), intent analysis (`medium` thinking), and datasheet extraction (`low` thinking) use `gemini-4-argon` (configurable via `MOTIONFIT_ARGON_MODEL`, with graceful fallback if needed).
   - Report prose generation uses `gemini-3.8-flash` (`MOTIONFIT_FLASH_MODEL`).
   - Schema/unit sanity routing uses `gemini-3.5-flash-lite` (`MOTIONFIT_LITE_MODEL`).
   - Uses `client.interactions.create(...)` with `previous_interaction_id`, `generation_config={"thinking_level": ...}`, and `response_format=[{"type": "text", "mime_type": "application/json", "schema": ...}]`. Never passes deprecated `temperature`, `top_p`, `top_k`, or `thinking_budget`.
4. **MuJoCo Composite Inertial Modeling (Critical Lesson #1)**:
   - **Quirk**: In MuJoCo MJCF, if a `<body>` includes an explicit `<inertial>` element, MuJoCo ignores `mass` attributes on all `<geom>` elements inside that same body.
   - **Fix**: `backend/model_builder.py` analytically computes the composite mass, center of mass (CoM), and diagonal inertia tensor of the joint actuator point mass (at $x = 0$) plus the hollow cylindrical link tube (spanning $x \in [0, L]$ with CoM at $L/2$) and writes the exact combined `<inertial>` tag. `<geom>` elements are purely visual/collision geometry. This guarantees exact agreement ($< 0.1\%$ error) between MuJoCo `mj_inverse` and the analytical static torque formula $g \sum m_i x_i$.
5. **4-DOF Base Singularity Avoidance (Critical Lesson #2)**:
   - **Quirk**: `q1 = atan2(p_y, p_x)` exhibits instantaneous $\pi$-radian flips if the hand path crosses near $(p_x, p_y) = (0, 0)$.
   - **Fix**: `backend/motion.py` maps human shoulder-relative coordinates into the robot's forward workspace with a safe forward offset ($x_0 > 0.20\text{ m}$), ensuring $r = \sqrt{p_x^2 + p_y^2}$ stays well away from the base yaw axis.
6. **Pass A (100 Hz) to Pass B (500 Hz) Controller Interpolation (Critical Lesson #3)**:
   - **Quirk**: Feeding 100 Hz zero-order-held reference angles/velocities into a 500 Hz high-gain PD controller ($K_p = 400 M_{jj}, K_d = 40 M_{jj}$) causes 100 Hz staircase chattering.
   - **Fix**: `backend/sim.py` linearly interpolates $q_{\text{ref}}(t)$, $\dot{q}_{\text{ref}}(t)$, and $\tau_{\text{ff}}(t)$ onto the 500 Hz simulation grid before computing $\tau_{\text{cmd}}$ and clamping to $\pm \tau_{\text{avail}}(\dot{q})$.
7. **Design Loop Token & Latency Optimization**:
   - Pass the confirmed, citation-tagged `catalog.json` into the design loop (`loop.py`) first turn and retain state via `previous_interaction_id`, keeping per-iteration latency well under the 40 s budget.
8. **CAD-Grade 3D Robot Visualization**:
   - The Three.js viewer (`frontend/src/viewer.ts`) builds a realistic articulated CAD-style robot arm inspired by CAE/Adams/MoveIt visualization: sculpted metallic/matte joint housings, chamfered link shells, base controller cabinet, ISO tool flange, plus translucent blue wireframe joint-axis cylinders and glowing green directional rotation arrows on every joint, with dynamic torque-ratio status lighting and ghost target trajectory overlay.
9. **MuJoCo `<compiler angle="radian"/>` Requirement (Critical Lesson #4)**:
   - **Quirk**: MuJoCo MJCF defaults to `angle="degree"` unless `<compiler angle="radian"/>` is explicitly present. Without it, `range="-2.97 2.97"` is interpreted as $\pm 2.97^\circ$ ($\approx \pm 0.05\text{ rad}$), activating huge joint-limit constraint torques ($>4,000\text{ N}\cdot\text{m}$) during `mj_inverse`.
   - **Fix**: Explicitly include `<compiler angle="radian"/>` in `backend/templates/arm4.xml.j2`.
10. **Pass B Distal Joint Damping Stability (Critical Lesson #5)**:
   - **Quirk**: Using a fixed nominal inertia ($0.05\text{ kg}\cdot\text{m}^2$) for $K_d$ across all joints exceeds the explicit Euler stability limit ($K_d \Delta t < 2 I_{jj}$) on the lightweight wrist joint J4 ($I_{44} \approx 0.0027\text{ kg}\cdot\text{m}^2$) at $\Delta t = 2\text{ ms}$, causing 500 Hz numerical oscillation.
   - **Fix**: Scale $K_d = 2.2 \sqrt{K_p \cdot M_{0,jj}}$ per joint using `model.dof_M0` and clamp to $0.35 M_{0,jj} / \Delta t$.
11. **4-DOF Analytic IK Over-Shoulder Plane (`r_ee < 0`) (Critical Lesson #6)**:
   - **Quirk**: When $q_2 > \pi/2$, the end-effector crosses behind the base $z$-axis ($r_{\text{ee}} < 0$), causing `atan2(y, x)` to shift by $\pi$.
   - **Fix**: `backend/kinematics.py` evaluates both forward ($+r$, $q_1 = \text{atan2}(y, x)$) and over-shoulder ($-r$, $q_1 = \text{atan2}(y, x) + \pi$) planar projections, achieving $0.0\text{ mm}$ IK round-trip error across all 1,000 random valid configurations.

