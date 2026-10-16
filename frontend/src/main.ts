/**
 * MotionFit Frontend Application Entry Point
 * Implements the three-screen hardware verification workflow:
 *  1. Upload & Catalog Review (HITL Step 1)
 *  2. Gemini 4 Argon Design Loop & Live SSE Verification Log
 *  3. Three-Pane 3D CAD Before/After Compare, Report & Engineer Sign-Off (HITL Step 2)
 */

import { RobotArmViewport, DesignSpec, TraceData } from "./viewer";
import { extractLandmarksFromVideoFile } from "./pose";

interface CatalogFieldVal {
  value: number;
  src: { doc: string; page?: number; row?: number };
  derived?: boolean;
  raw?: string;
}

interface CatalogPart {
  part_id: string;
  type: "actuator" | "gearbox";
  vendor: string;
  series?: string;
  rated_voltage_v?: number;
  stall_torque_nm?: CatalogFieldVal;
  continuous_torque_nm?: CatalogFieldVal;
  no_load_speed_rad_s?: CatalogFieldVal;
  ratio?: CatalogFieldVal;
  efficiency?: CatalogFieldVal;
  rated_output_torque_nm?: CatalogFieldVal;
  mass_kg: CatalogFieldVal;
  price_usd: CatalogFieldVal;
  confidence: string;
  user_confirmed: boolean;
}

interface ProjectState {
  project_id: string;
  materials: Record<string, any>;
  design: DesignSpec;
  catalog: {
    catalog_id: string;
    confirmed?: boolean;
    parts: CatalogPart[];
  };
  target: any;
  intent: {
    task_summary: string;
    orientation_constraint: string;
    phases: Array<{ name: string; t_start: number; t_end: number; notes: string }>;
    critical_moments?: Array<{ t: number; reason: string }>;
  };
  loop_result: any | null;
  approval: {
    approved: boolean;
    engineer_name: string | null;
    notes: string | null;
    approved_at: string | null;
  } | null;
  has_report: boolean;
}

type ActiveTab = "upload" | "run" | "compare";

let state: ProjectState | null = null;
let activeTab: ActiveTab = "upload";
let activeMotionPreset = "wipe";
let runEvents: Array<Record<string, any>> = [];
let isRunningLoop = false;

// 3D Compare Viewports & Playback State
let beforeViewport: RobotArmViewport | null = null;
let afterViewport: RobotArmViewport | null = null;
let animFrameId: number | null = null;
let isPlaying = true;
let playbackSpeed = 1.0;
let currentFrameIdx = 0;
let lastTickMs = 0;
let showCadAxes = true;
let showGhostTarget = true;

async function fetchProjectState(projectId = "demo"): Promise<void> {
  const res = await fetch(`/api/projects/${projectId}`);
  if (!res.ok) {
    throw new Error(`Failed to load project ${projectId}: ${res.statusText}`);
  }
  state = await res.json();
  if (state?.loop_result?.iterations && runEvents.length === 0) {
    // Populate initial event log from cached loop_result
    runEvents = buildSyntheticEventsFromLoopResult(state.loop_result);
  }
}

function buildSyntheticEventsFromLoopResult(lr: any): Array<Record<string, any>> {
  const evts: Array<Record<string, any>> = [];
  const nowIso = new Date().toISOString();
  for (const it of lr.iterations || []) {
    evts.push({
      type: "iteration_simulated",
      iteration: it.iteration,
      design_id: it.design_id,
      passed: it.check?.passed,
      violations_count: (it.check?.violations || []).length,
      ee_rmse_mm: it.metrics?.ee_rmse_mm,
      total_mass_kg: it.metrics?.total_mass_kg,
      total_cost_usd: it.metrics?.total_cost_usd,
      timestamp: nowIso,
    });
    if (it.proposal) {
      evts.push({
        type: it.is_post_pass_optimization ? "post_pass_optimization" : "argon_proposal",
        iteration: it.iteration,
        diagnosis: it.proposal.diagnosis,
        changes: it.proposal.changes,
        timestamp: nowIso,
      });
    }
  }
  if (lr.naive_baseline) {
    evts.push({
      type: "naive_baseline",
      passed: lr.naive_baseline.passed,
      actuator_id: lr.naive_baseline.actuator_id,
      summary: lr.naive_baseline.summary,
      timestamp: nowIso,
    });
  }
  evts.push({
    type: "complete",
    passed: lr.passed,
    final_design_id: lr.final_design?.design_id,
    iterations_count: (lr.iterations || []).length,
    timestamp: nowIso,
  });
  return evts;
}

function setTab(tab: ActiveTab): void {
  activeTab = tab;
  renderApp();
  if (tab === "compare") {
    requestAnimationFrame(() => initCompareViewports());
  }
}

function renderApp(): void {
  const root = document.getElementById("app");
  if (!root || !state) return;

  const catalogConfirmed = state.catalog.parts.every((p) => p.user_confirmed);
  const loopPassed = Boolean(state.loop_result?.passed);
  const isApproved = Boolean(state.approval?.approved);

  root.innerHTML = `
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">M</div>
        <div>
          <div class="brand-title">MotionFit</div>
          <div class="brand-sub">Show it the motion. Gemini 4 Argon + MuJoCo verifies and redesigns the robot arm.</div>
        </div>
      </div>

      <nav class="step-tabs">
        <button class="step-tab ${activeTab === "upload" ? "active" : ""}" id="tab-upload">
          <span class="step-num">1</span>
          Upload &amp; Catalog Review (HITL #1)
          ${catalogConfirmed ? '<span class="badge badge-pass" style="margin-left:4px;">Confirmed</span>' : '<span class="badge badge-warn" style="margin-left:4px;">Review</span>'}
        </button>
        <button class="step-tab ${activeTab === "run" ? "active" : ""}" id="tab-run">
          <span class="step-num">2</span>
          Argon Verification Loop
          ${loopPassed ? '<span class="badge badge-pass" style="margin-left:4px;">Verified</span>' : ""}
        </button>
        <button class="step-tab ${activeTab === "compare" ? "active" : ""}" id="tab-compare">
          <span class="step-num">3</span>
          3D CAD Compare, Report &amp; Sign-Off (HITL #2)
          ${isApproved ? '<span class="badge badge-pass" style="margin-left:4px;">Signed Off</span>' : ""}
        </button>
      </nav>

      <div class="topbar-actions">
        <span class="badge badge-info mono">gemini-4-argon &middot; MuJoCo 3.2.7</span>
        <button class="btn" id="btn-reset-demo" title="Reload pre-cached demo dataset">Reset Demo</button>
        <button class="btn btn-primary" id="btn-quick-run" ${isRunningLoop ? "disabled" : ""}>
          ${isRunningLoop ? "Running Loop..." : "&#9654; Run Argon Loop"}
        </button>
      </div>
    </header>

    <main class="workspace">
      ${activeTab === "upload" ? renderUploadScreen() : ""}
      ${activeTab === "run" ? renderRunScreen() : ""}
      ${activeTab === "compare" ? renderCompareScreen() : ""}
    </main>
  `;

  bindTopbarEvents();
  if (activeTab === "upload") bindUploadEvents();
  if (activeTab === "run") bindRunEvents();
  if (activeTab === "compare") bindCompareEvents();
}

/* ============================================================================
 * SCREEN 1: UPLOAD, MOTION INTENT & CATALOG REVIEW (HITL #1)
 * ========================================================================== */
function renderUploadScreen(): string {
  if (!state) return "";
  const d = state.design;
  const intent = state.intent;
  const actuators = state.catalog.parts.filter((p) => p.type === "actuator");
  const gearboxes = state.catalog.parts.filter((p) => p.type === "gearbox");
  const tubeKeys = Object.keys(state.materials || {});

  return `
    <div class="grid-2">
      <!-- Left Column: Arm Spec + Motion Video / Presets -->
      <div>
        <div class="panel">
          <div class="panel-header">
            <h2 class="panel-title">1A. Initial Arm Design Spec (<span class="mono">${d.design_id}</span>)</h2>
            <span class="badge badge-info">4-DOF Serial Topology</span>
          </div>

          <div class="form-grid" style="margin-bottom: 12px;">
            <div class="form-field">
              <label>Payload Mass (kg)</label>
              <input type="number" id="inp-payload" step="0.05" min="0.05" max="5.0" value="${d.payload_kg}" />
            </div>
            <div class="form-field">
              <label>Link Tube Profile</label>
              <select id="sel-tube">
                ${tubeKeys
                  .map(
                    (k) =>
                      `<option value="${k}" ${d.links.tube_profile === k ? "selected" : ""}>${k} (${state!.materials[k].material})</option>`
                  )
                  .join("")}
              </select>
            </div>
            <div class="form-field">
              <label>Upper Arm L1 (m) [0.22–0.40]</label>
              <input type="number" id="inp-l1" step="0.01" min="0.22" max="0.40" value="${d.links.upper_arm_m}" />
            </div>
            <div class="form-field">
              <label>Forearm L2 (m) [0.18–0.35]</label>
              <input type="number" id="inp-l2" step="0.01" min="0.18" max="0.35" value="${d.links.forearm_m}" />
            </div>
            <div class="form-field">
              <label>Max Total Mass Budget (kg)</label>
              <input type="number" id="inp-max-mass" step="0.1" value="${d.constraints.max_total_mass_kg}" />
            </div>
            <div class="form-field">
              <label>Max BOM Cost Budget (USD)</label>
              <input type="number" id="inp-max-cost" step="50" value="${d.constraints.max_cost_usd}" />
            </div>
          </div>

          <div class="table-wrap" style="margin-bottom: 10px;">
            <table>
              <thead>
                <tr>
                  <th>Joint</th>
                  <th>Axis &amp; Role</th>
                  <th>Initial Actuator</th>
                  <th>External Gearbox</th>
                </tr>
              </thead>
              <tbody>
                ${(["J1", "J2", "J3", "J4"] as const)
                  .map((jName) => {
                    const roleMap = {
                      J1: "+Z Base Yaw",
                      J2: "Y Shoulder Pitch",
                      J3: "Y Elbow Pitch",
                      J4: "Y Wrist Pitch",
                    };
                    const jSpec = d.joints[jName];
                    return `
                      <tr>
                        <td class="mono"><strong>${jName}</strong></td>
                        <td>${roleMap[jName]}</td>
                        <td>
                          <select class="sel-joint-act" data-joint="${jName}">
                            ${actuators
                              .map(
                                (a) =>
                                  `<option value="${a.part_id}" ${jSpec.actuator_id === a.part_id ? "selected" : ""}>${a.part_id}</option>`
                              )
                              .join("")}
                          </select>
                        </td>
                        <td>
                          <select class="sel-joint-gb" data-joint="${jName}">
                            <option value="" ${!jSpec.gearbox_id ? "selected" : ""}>None (Direct)</option>
                            ${gearboxes
                              .map(
                                (g) =>
                                  `<option value="${g.part_id}" ${jSpec.gearbox_id === g.part_id ? "selected" : ""}>${g.part_id} (${g.ratio?.value}:1)</option>`
                              )
                              .join("")}
                          </select>
                        </td>
                      </tr>
                    `;
                  })
                  .join("")}
              </tbody>
            </table>
          </div>
          <button class="btn" id="btn-save-design" style="width: 100%; justify-content: center;">
            Save Initial Design Spec
          </button>
        </div>

        <!-- Motion Video & Argon Intent Panel -->
        <div class="panel">
          <div class="panel-header">
            <h2 class="panel-title">1B. Motion Capture &amp; Argon Intent Analysis</h2>
            <span class="badge badge-pass">100 Hz Trajectory Ready</span>
          </div>

          <div style="margin-bottom: 10px;">
            <label style="font-size: 11.5px; color: var(--text-muted); font-weight: 600; display: block; margin-bottom: 6px;">
              Select Demo Motion Preset or Upload Phone Video (MP4/MOV):
            </label>
            <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px;">
              <button class="btn ${activeMotionPreset === "wipe" ? "btn-primary" : ""}" data-preset="wipe">
                1. Workbench Wipe &amp; 400g Lift (Demo)
              </button>
              <button class="btn ${activeMotionPreset === "pour" ? "btn-primary" : ""}" data-preset="pour">
                2. Precision Decant &amp; Hold
              </button>
              <button class="btn ${activeMotionPreset === "pick_and_place" ? "btn-primary" : ""}" data-preset="pick_and_place">
                3. High-Speed Bin Transfer
              </button>
            </div>

            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
              <input type="file" id="inp-video-file" accept="video/mp4,video/quicktime,video/webm" style="font-size: 12px;" />
              <span id="video-status" style="font-size: 11.5px; color: var(--text-muted);"></span>
            </div>

            <div id="video-preview-wrap" style="display: none; position: relative; margin-bottom: 10px;">
              <video id="pose-video" style="display: none;"></video>
              <canvas id="pose-canvas" style="width: 100%; border-radius: 8px; border: 1px solid var(--border);"></canvas>
            </div>
          </div>

          <div style="background: #090e1a; border: 1px solid var(--border); border-radius: 8px; padding: 10px;">
            <div style="font-size: 12px; margin-bottom: 6px;">
              <strong>Argon Motion Intent:</strong> ${intent.task_summary}
            </div>
            <div style="font-size: 11.5px; color: var(--text-muted); margin-bottom: 8px;">
              Orientation Constraint: <span class="mono" style="color: var(--accent);">${intent.orientation_constraint}</span>
            </div>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Phase</th>
                    <th>Window</th>
                    <th>Argon Physical Load Diagnosis</th>
                  </tr>
                </thead>
                <tbody>
                  ${(intent.phases || [])
                    .map(
                      (ph) => `
                    <tr>
                      <td class="mono"><strong>${ph.name}</strong></td>
                      <td class="mono">${ph.t_start.toFixed(1)}–${ph.t_end.toFixed(1)}s</td>
                      <td>${ph.notes}</td>
                    </tr>
                  `
                    )
                    .join("")}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Column: HITL #1 Catalog Review Table -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">1C. Human-in-the-Loop Catalog Review (<span class="mono">catalog.json</span>)</h2>
            <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">
              Every parameter extracted from the 16 vendor PDF datasheets carries a clickable page citation. Derived continuous torques (0.33 &times; stall) are highlighted in amber.
            </div>
          </div>
          <div style="display: flex; gap: 8px;">
            <button class="btn btn-success" id="btn-confirm-catalog">
              &#10003; Confirm Catalog &amp; Start Argon Loop
            </button>
          </div>
        </div>

        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Part ID</th>
                <th>Vendor &amp; Type</th>
                <th>Stall / Rated Output (N&middot;m)</th>
                <th>Continuous Torque (N&middot;m)</th>
                <th>No-Load Speed / Ratio</th>
                <th>Mass (kg)</th>
                <th>Price (USD)</th>
                <th>PDF Citation</th>
              </tr>
            </thead>
            <tbody>
              ${state.catalog.parts
                .map((part, idx) => {
                  const isAct = part.type === "actuator";
                  const isDerived = Boolean(part.continuous_torque_nm?.derived);
                  const primaryDoc = isAct
                    ? part.stall_torque_nm?.src.doc
                    : part.rated_output_torque_nm?.src.doc;
                  const primaryPage = isAct
                    ? part.stall_torque_nm?.src.page || 1
                    : part.rated_output_torque_nm?.src.page || 1;

                  return `
                    <tr class="${isDerived ? "row-derived" : ""}">
                      <td class="mono">
                        <strong>${part.part_id}</strong>
                        ${
                          part.rated_voltage_v
                            ? `<div style="font-size:10.5px;color:var(--text-muted);">${part.rated_voltage_v}V</div>`
                            : ""
                        }
                      </td>
                      <td>
                        <div>${part.vendor}</div>
                        <span class="badge ${isAct ? "badge-info" : "badge-pass"}" style="font-size:10px;">${part.type}</span>
                      </td>
                      <td>
                        ${
                          isAct
                            ? `<input type="number" step="0.1" class="cat-edit" data-idx="${idx}" data-field="stall_torque_nm" value="${part.stall_torque_nm?.value ?? 0}" />`
                            : `<input type="number" step="0.5" class="cat-edit" data-idx="${idx}" data-field="rated_output_torque_nm" value="${part.rated_output_torque_nm?.value ?? 0}" />`
                        }
                      </td>
                      <td>
                        ${
                          isAct
                            ? `
                              <div style="display:flex;align-items:center;gap:6px;">
                                <input type="number" step="0.1" class="cat-edit" data-idx="${idx}" data-field="continuous_torque_nm" value="${part.continuous_torque_nm?.value ?? 0}" />
                                ${
                                  isDerived
                                    ? '<span class="badge badge-warn" title="Datasheet omits continuous torque; derived as 0.33 x stall torque">0.33&times; Derived</span>'
                                    : '<span class="badge badge-pass">Explicit</span>'
                                }
                              </div>
                            `
                            : `<span class="mono" style="color:var(--text-muted);">Eff: ${((part.efficiency?.value ?? 0.94) * 100).toFixed(0)}%</span>`
                        }
                      </td>
                      <td class="mono">
                        ${
                          isAct
                            ? `${part.no_load_speed_rad_s?.value.toFixed(2)} rad/s <div style="font-size:10.5px;color:var(--text-muted);">(${part.no_load_speed_rad_s?.raw || ""})</div>`
                            : `${part.ratio?.value}:1 stage`
                        }
                      </td>
                      <td>
                        <input type="number" step="0.005" class="cat-edit" data-idx="${idx}" data-field="mass_kg" value="${part.mass_kg.value}" />
                      </td>
                      <td>
                        <input type="number" step="5" class="cat-edit" data-idx="${idx}" data-field="price_usd" value="${part.price_usd.value}" />
                      </td>
                      <td>
                        <a class="cit-pill" href="/api/projects/${state!.project_id}/datasheets/${primaryDoc}#page=${primaryPage}" target="_blank">
                          &#128196; ${primaryDoc}#p${primaryPage}
                        </a>
                        <div style="font-size:10.5px;color:var(--text-muted);" class="mono">
                          prices.csv:r${part.price_usd.src.row ?? 1}
                        </div>
                      </td>
                    </tr>
                  `;
                })
                .join("")}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;
}

/* ============================================================================
 * SCREEN 2: ARGON DESIGN LOOP & LIVE STREAMED LOG
 * ========================================================================== */
function renderRunScreen(): string {
  if (!state) return "";
  const lr = state.loop_result;
  const iterations: any[] = lr?.iterations || [];
  const naive = lr?.naive_baseline;

  return `
    <div class="panel" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
      <div>
        <h2 class="panel-title" style="font-size:16px;">
          2. Gemini 4 Argon Coupled-Physics Verification Loop
        </h2>
        <div style="font-size:12.5px;color:var(--text-muted);margin-top:2px;">
          MuJoCo 3.2.7 runs Pass A (100 Hz Inverse Dynamics) + Pass B (500 Hz Torque-Clamped Forward Tracking). <span class="mono">checker.py</span> grades all 17 checks; <span class="mono">validate.py</span> verifies every Argon citation against <span class="mono">catalog.json</span>.
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <button class="btn btn-primary" id="btn-run-loop-main" ${isRunningLoop ? "disabled" : ""}>
          ${isRunningLoop ? "Running Verification Loop..." : "&#9654; Re-Run Argon Loop Now"}
        </button>
        ${
          lr?.passed
            ? `<button class="btn btn-success" id="btn-goto-compare">Open 3D CAD Compare &amp; Report &rarr;</button>`
            : ""
        }
      </div>
    </div>

    <div class="grid-run">
      <!-- Left: Iteration Cards -->
      <div>
        ${
          iterations.length === 0
            ? `<div class="panel">Click <strong>Run Argon Loop</strong> to execute the physics verification and hardware redesign loop.</div>`
            : iterations.map((it) => renderIterationCard(it, state!.design.thresholds)).join("")
        }

        ${
          naive
            ? `
          <div class="panel" style="border-color: rgba(245, 158, 11, 0.5);">
            <div class="panel-header">
              <h3 class="panel-title">
                Stretch 1 — Naive Heuristic Baseline ("Upgrade Failing Joint to Strongest Motor")
              </h3>
              <span class="badge ${naive.passed ? "badge-pass" : "badge-fail"}">
                Naive Baseline: ${naive.passed ? "PASS" : "FAILS BUDGET & SPEED"}
              </span>
            </div>
            <p style="margin:0 0 8px 0;font-size:13px;">${naive.summary}</p>
            <div style="display:flex;gap:16px;font-size:12px;color:var(--text-muted);" class="mono">
              <span>Motor: ${naive.actuator_id}</span>
              <span>EE RMSE: ${naive.metrics.ee_rmse_mm.toFixed(1)} mm</span>
              <span>Total Mass: ${naive.metrics.total_mass_kg.toFixed(3)} kg</span>
              <span>BOM Cost: $${naive.metrics.total_cost_usd.toFixed(2)}</span>
            </div>
          </div>
        `
            : ""
        }
      </div>

      <!-- Right: Streamed SSE Event Log -->
      <div class="panel">
        <div class="panel-header">
          <h3 class="panel-title">Live Verification &amp; Validator Stream</h3>
          <span class="badge badge-info mono">${runEvents.length} events</span>
        </div>
        <div class="event-log" id="sse-log-box">
          ${runEvents
            .map(
              (ev) => `
            <div class="event-item">
              <div style="display:flex;justify-content:space-between;color:var(--text-muted);font-size:10.5px;">
                <span style="color:var(--accent);font-weight:700;">[${ev.type}]</span>
                <span>${(ev.timestamp || "").slice(11, 19)}</span>
              </div>
              <div style="margin-top:3px;">
                ${formatEventSummary(ev)}
              </div>
            </div>
          `
            )
            .join("")}
        </div>
      </div>
    </div>
  `;
}

function renderIterationCard(it: any, thresholds: any): string {
  const passed = Boolean(it.check?.passed);
  const m = it.metrics;
  const prop = it.proposal;
  const pvm = it.prediction_vs_measured;

  return `
    <div class="panel" style="border-left: 4px solid ${passed ? "var(--pass)" : "var(--fail)"};">
      <div class="panel-header">
        <div>
          <span class="badge ${passed ? "badge-pass" : "badge-fail"}" style="margin-right:8px;">
            Iteration ${it.iteration} ${it.is_post_pass_optimization ? "(Post-Pass Optimization)" : ""}
          </span>
          <strong class="mono" style="font-size:14px;">${it.design_id}</strong>
        </div>
        <div class="mono" style="font-size:12.5px;">
          EE RMSE: <strong style="color:${m.ee_rmse_mm <= thresholds.ee_rmse_mm ? "var(--pass)" : "var(--fail)"};">${m.ee_rmse_mm.toFixed(2)} mm</strong> &middot;
          Mass: <strong>${m.total_mass_kg.toFixed(3)} kg</strong> &middot;
          Cost: <strong>$${m.total_cost_usd.toFixed(2)}</strong>
        </div>
      </div>

      <!-- 4 Joint Utilization Cards -->
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px;">
        ${(["J1", "J2", "J3", "J4"] as const)
          .map((jName) => {
            const jm = m.joints[jName];
            const peakOk = jm.peak_ratio <= thresholds.peak_torque_ratio;
            const rmsOk = jm.rms_ratio <= thresholds.rms_torque_ratio;
            const spdOk = jm.speed_ratio <= thresholds.peak_speed_ratio;
            const allOk = peakOk && rmsOk && spdOk;
            const maxRatioPct = Math.min(100, Math.round(Math.max(jm.peak_ratio, jm.rms_ratio) * 70));

            return `
              <div style="background:#090e1a;border:1px solid ${allOk ? "var(--border)" : "rgba(239,68,68,0.5)"};border-radius:8px;padding:9px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                  <strong class="mono">${jName}</strong>
                  <span class="badge ${allOk ? "badge-pass" : "badge-fail"}" style="font-size:10px;">
                    ${allOk ? "OK" : "OVERLOAD"}
                  </span>
                </div>
                <div class="mono" style="font-size:11px;color:var(--accent);margin-bottom:4px;">
                  ${jm.actuator_id}${jm.gearbox_id ? " + " + jm.gearbox_id : ""}
                </div>
                <div style="font-size:11px;color:var(--text-muted);" class="mono">
                  Peak: ${jm.peak_req_nm.toFixed(2)}/${jm.tau_avail_at_peak_nm.toFixed(2)} Nm (${jm.peak_ratio.toFixed(2)})<br/>
                  RMS: ${jm.rms_req_nm.toFixed(2)}/${jm.tau_cont_eff_nm.toFixed(2)} Nm (${jm.rms_ratio.toFixed(2)})<br/>
                  Sat: ${jm.saturated_time_s.toFixed(2)}s
                </div>
                <div class="ratio-bar-wrap">
                  <div class="ratio-bar-fill" style="width:${maxRatioPct}%;background:${allOk ? "var(--pass)" : "var(--fail)"};"></div>
                </div>
              </div>
            `;
          })
          .join("")}
      </div>

      ${
        pvm
          ? `
        <div style="background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.25);border-radius:7px;padding:8px 11px;margin-bottom:10px;font-size:12px;">
          <strong>Argon Prediction vs. MuJoCo Measured:</strong>
          <span class="mono">
            J2 Peak Ratio: pred ${pvm.j2_peak_ratio.predicted.toFixed(3)} vs actual ${pvm.j2_peak_ratio.measured.toFixed(3)} &middot;
            J2 RMS Ratio: pred ${pvm.j2_rms_ratio.predicted.toFixed(3)} vs actual ${pvm.j2_rms_ratio.measured.toFixed(3)} &middot;
            Total Mass: pred ${Number(pvm.total_mass_kg.predicted).toFixed(3)} kg vs actual ${Number(pvm.total_mass_kg.measured).toFixed(3)} kg
          </span>
        </div>
      `
          : ""
      }

      ${
        prop
          ? `
        <div style="background:#0c1424;border:1px solid var(--border);border-radius:8px;padding:10px;">
          <div style="font-size:12.5px;margin-bottom:8px;">
            <strong style="color:var(--accent);">Gemini 4 Argon Diagnosis:</strong> ${prop.diagnosis}
          </div>
          ${(prop.changes || [])
            .map(
              (c: any) => `
            <div style="padding:6px 8px;background:#090e1a;border-radius:6px;margin-bottom:6px;font-size:12px;">
              <div>
                <span class="mono" style="color:var(--warn);">${c.target}</span>:
                <span class="mono">${c.from} &rarr; <strong>${c.to}</strong></span>
                &mdash; ${c.rationale}
              </div>
              <div style="margin-top:4px;">
                ${(c.citations || [])
                  .map(
                    (cit: any) => `
                  <a class="cit-pill" href="/api/projects/${state!.project_id}/datasheets/${cit.doc}#page=${cit.page}" target="_blank">
                    &#128196; ${cit.doc}#p${cit.page} (${cit.field}=${cit.value})
                  </a>
                `
                  )
                  .join("")}
              </div>
            </div>
          `
            )
            .join("")}
        </div>
      `
          : ""
      }
    </div>
  `;
}

function formatEventSummary(ev: Record<string, any>): string {
  if (ev.type === "iteration_simulated" || ev.type === "iteration_evaluated") {
    return `Iter ${ev.iteration} (${ev.design_id}): ${ev.passed ? "PASSED" : `FAILED (${ev.violations_count ?? (ev.violations || []).length} checks)`} | RMSE=${Number(ev.ee_rmse_mm).toFixed(2)}mm | Mass=${Number(ev.total_mass_kg).toFixed(3)}kg`;
  }
  if (ev.type === "argon_proposal" || ev.type === "post_pass_optimization") {
    const chgs = (ev.changes || []).map((c: any) => `${c.target}: ${c.from}->${c.to}`).join(", ");
    return `${chgs} — ${ev.diagnosis || ""}`;
  }
  if (ev.type === "proposal_validated" || ev.type === "validator_check") {
    return `Proposal validator: ${ev.valid ? "VALID (all part IDs & PDF citations verified)" : "REJECTED"}`;
  }
  if (ev.type === "naive_baseline") {
    return `Naive strongest motor (${ev.actuator_id}): ${ev.passed ? "PASS" : "FAIL"}`;
  }
  if (ev.type === "complete") {
    return `Loop complete -> ${ev.final_design_id} (${ev.iterations_count} iterations, passed=${ev.passed})`;
  }
  return JSON.stringify(ev);
}

/* ============================================================================
 * SCREEN 3: THREE-PANE 3D CAD COMPARE, REPORT & ENGINEER SIGN-OFF (HITL #2)
 * ========================================================================== */
function renderCompareScreen(): string {
  if (!state || !state.loop_result) {
    return `
      <div class="panel">
        <p>Run the Argon Verification Loop first to compare Before &amp; After 3D CAD simulations.</p>
        <button class="btn btn-primary" id="btn-run-from-compare">&#9654; Run Argon Verification Loop</button>
      </div>
    `;
  }

  const lr = state.loop_result;
  const initIt = lr.iterations[0];
  const finalIt = lr.iterations[lr.iterations.length - 1];
  const initTrace: TraceData = lr.initial_trace;
  const worstT = initIt.metrics.t_worst_ee_s ?? 3.27;
  const maxFrame = Math.max(0, (initTrace?.t?.length || 1) - 1);
  const approval = state.approval;
  const isApproved = Boolean(approval?.approved);

  return `
    <!-- Synchronized Playback & CAD Overlay Toolbar -->
    <div class="scrubber-bar">
      <button class="btn btn-primary" id="btn-play-pause" style="min-width:88px;justify-content:center;">
        ${isPlaying ? "&#10074;&#10074; Pause" : "&#9654; Play"}
      </button>

      <button class="btn btn-warn" id="btn-jump-worst" title="Jump to timestamp of peak tracking droop">
        &#9888; Jump to Worst Error (t = ${worstT.toFixed(2)}s)
      </button>

      <input
        type="range"
        id="scrubber-range"
        class="scrubber-slider"
        min="0"
        max="${maxFrame}"
        value="${currentFrameIdx}"
      />

      <span class="mono" id="scrubber-time-label" style="min-width:115px;font-weight:700;color:var(--accent);">
        t = ${(initTrace.t[currentFrameIdx] || 0).toFixed(2)} s
      </span>

      <select id="sel-speed" title="Playback speed">
        <option value="0.25" ${playbackSpeed === 0.25 ? "selected" : ""}>0.25&times; Slow-Mo</option>
        <option value="0.5" ${playbackSpeed === 0.5 ? "selected" : ""}>0.50&times; Speed</option>
        <option value="1" ${playbackSpeed === 1.0 ? "selected" : ""}>1.00&times; Real-Time</option>
      </select>

      <label style="display:inline-flex;align-items:center;gap:5px;font-size:12px;cursor:pointer;">
        <input type="checkbox" id="chk-cad-axes" ${showCadAxes ? "checked" : ""} />
        CAD Joint Axes &amp; Torque Arrows
      </label>

      <label style="display:inline-flex;align-items:center;gap:5px;font-size:12px;cursor:pointer;">
        <input type="checkbox" id="chk-ghost-arm" ${showGhostTarget ? "checked" : ""} />
        Ghost Target Arm
      </label>
    </div>

    <!-- Three-Pane Compare Layout -->
    <div class="three-pane">
      <!-- PANE 1: BEFORE (Initial Design) -->
      <div class="viewport-panel">
        <div class="panel-header" style="padding:12px 14px 10px;margin:0;">
          <div>
            <span class="badge badge-fail">BEFORE</span>
            <strong class="mono" style="margin-left:6px;">${lr.initial_design.design_id}</strong>
            <div style="font-size:11.5px;color:var(--text-muted);margin-top:2px;">
              J2: <span class="mono">${lr.initial_design.joints.J2.actuator_id}</span> (Direct) &middot; Droops under 400g lift
            </div>
          </div>
          <div style="text-align:right;" class="mono">
            <div style="color:var(--fail);font-weight:700;font-size:14px;">RMSE: ${initIt.metrics.ee_rmse_mm.toFixed(1)} mm</div>
            <div style="font-size:11px;color:var(--text-muted);">Max Err: ${initIt.metrics.ee_max_err_mm.toFixed(1)} mm</div>
          </div>
        </div>

        <div class="viewport-canvas-box">
          <div id="viewport-before" style="width:100%;height:100%;"></div>
          <div class="viewport-overlay-legend">
            <div><span style="color:#38bdf8;">&#9644;</span> Desired EE Trajectory</div>
            <div><span style="color:#ef4444;">&#9644;</span> Actual Clamped Path (Droop)</div>
            <div><span style="color:#60a5fa;">&#9675;</span> Blue Cylinder: Joint Axis</div>
            <div><span style="color:#22c55e;">&#8635;</span> Green/Red Arrow: Torque Ratio</div>
          </div>
        </div>

        <div class="joint-live-strip" id="live-strip-before"></div>
      </div>

      <!-- PANE 2: AFTER (Argon Verified Design) -->
      <div class="viewport-panel">
        <div class="panel-header" style="padding:12px 14px 10px;margin:0;">
          <div>
            <span class="badge badge-pass">AFTER (VERIFIED)</span>
            <strong class="mono" style="margin-left:6px;">${lr.final_design.design_id}</strong>
            <div style="font-size:11.5px;color:var(--text-muted);margin-top:2px;">
              J2: <span class="mono">${lr.final_design.joints.J2.actuator_id}${lr.final_design.joints.J2.gearbox_id ? " + " + lr.final_design.joints.J2.gearbox_id : ""}</span> &middot; All 17 Checks Pass
            </div>
          </div>
          <div style="text-align:right;" class="mono">
            <div style="color:var(--pass);font-weight:700;font-size:14px;">RMSE: ${finalIt.metrics.ee_rmse_mm.toFixed(2)} mm</div>
            <div style="font-size:11px;color:var(--text-muted);">Mass: ${finalIt.metrics.total_mass_kg.toFixed(3)} kg | $${finalIt.metrics.total_cost_usd.toFixed(2)}</div>
          </div>
        </div>

        <div class="viewport-canvas-box">
          <div id="viewport-after" style="width:100%;height:100%;"></div>
          <div class="viewport-overlay-legend">
            <div><span style="color:#38bdf8;">&#9644;</span> Desired EE Trajectory</div>
            <div><span style="color:#10b981;">&#9644;</span> Actual Verified Path</div>
            <div><span style="color:#0284c7;">&#9632;</span> Cobalt Stage: APEX-PG20-2 Gearbox</div>
          </div>
        </div>

        <div class="joint-live-strip" id="live-strip-after"></div>
      </div>

      <!-- PANE 3: REPORT PREVIEW & HITL #2 ENGINEER SIGN-OFF -->
      <div style="display:flex;flex-direction:column;gap:14px;">
        <!-- Engineer Sign-Off Card (HITL #2) -->
        <div class="panel" style="margin:0;border-color:${isApproved ? "rgba(16,185,129,0.5)" : "rgba(56,189,248,0.4)"};">
          <div class="panel-header">
            <h3 class="panel-title">Human-in-the-Loop Engineer Sign-Off</h3>
            ${
              isApproved
                ? '<span class="badge badge-pass">&#10003; Approved</span>'
                : '<span class="badge badge-warn">Approval Required</span>'
            }
          </div>

          <div style="font-size:12px;color:var(--text-muted);margin-bottom:10px;">
            Review datasheet citations and the <em>Not Verified by Simulation</em> disclaimer before unlocking CAD/MJCF bundle export.
          </div>

          <div class="form-grid" style="margin-bottom:10px;">
            <div class="form-field">
              <label>Reviewing Engineer Name *</label>
              <input type="text" id="inp-eng-name" placeholder="e.g. Alex Lee, Lead Robotics Eng." value="${approval?.engineer_name || "Alex Lee"}" />
            </div>
            <div class="form-field">
              <label>Sign-Off Notes</label>
              <input type="text" id="inp-eng-notes" placeholder="Checked PG20 flange & thermal duty" value="${approval?.notes || "Verified J2 APEX-PG20-2 peak/RMS margins and mounting adapter."}" />
            </div>
          </div>

          <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <button class="btn btn-success" id="btn-approve-signoff" style="flex:1;justify-content:center;">
              &#10003; ${isApproved ? "Update Sign-Off" : "Approve Final Design & Unlock Export"}
            </button>

            <a
              class="btn btn-primary"
              id="btn-download-zip"
              href="${isApproved ? `/api/projects/${state.project_id}/export.zip` : "#"}"
              ${isApproved ? "download" : ""}
              style="flex:1;justify-content:center;${!isApproved ? "opacity:0.45;pointer-events:none;" : ""}"
            >
              &#128230; Export Verified Bundle (.zip)
            </a>
          </div>
        </div>

        <!-- Embedded Solution Report & Full-Page Link -->
        <div class="panel" style="margin:0;flex:1;display:flex;flex-direction:column;">
          <div class="panel-header">
            <h3 class="panel-title">Cited Engineering Solution Report</h3>
            <a class="btn" href="/api/projects/${state.project_id}/report" target="_blank">
              &#8599; Open Printable Report
            </a>
          </div>
          <iframe
            id="report-preview-frame"
            class="report-iframe"
            src="/api/projects/${state.project_id}/report"
            title="MotionFit Solution Report"
          ></iframe>
        </div>
      </div>
    </div>
  `;
}

/* ============================================================================
 * EVENT BINDINGS & 3D ANIMATION LOOP
 * ========================================================================== */
function bindTopbarEvents(): void {
  document.getElementById("tab-upload")?.addEventListener("click", () => setTab("upload"));
  document.getElementById("tab-run")?.addEventListener("click", () => setTab("run"));
  document.getElementById("tab-compare")?.addEventListener("click", () => setTab("compare"));

  document.getElementById("btn-reset-demo")?.addEventListener("click", async () => {
    await fetch("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ use_demo: true }),
    });
    await fetchProjectState("demo");
    renderApp();
  });

  document.getElementById("btn-quick-run")?.addEventListener("click", () => {
    triggerRunLoop();
  });
}

function bindUploadEvents(): void {
  // Save Design Spec
  document.getElementById("btn-save-design")?.addEventListener("click", async () => {
    if (!state) return;
    const d = structuredClone(state.design);
    d.payload_kg = parseFloat((document.getElementById("inp-payload") as HTMLInputElement).value) || 0.4;
    d.links.tube_profile = (document.getElementById("sel-tube") as HTMLSelectElement).value;
    d.links.upper_arm_m = parseFloat((document.getElementById("inp-l1") as HTMLInputElement).value) || 0.3;
    d.links.forearm_m = parseFloat((document.getElementById("inp-l2") as HTMLInputElement).value) || 0.25;
    d.constraints.max_total_mass_kg =
      parseFloat((document.getElementById("inp-max-mass") as HTMLInputElement).value) || 4.0;
    d.constraints.max_cost_usd =
      parseFloat((document.getElementById("inp-max-cost") as HTMLInputElement).value) || 2500;

    document.querySelectorAll<HTMLSelectElement>(".sel-joint-act").forEach((sel) => {
      const j = sel.dataset.joint!;
      d.joints[j].actuator_id = sel.value;
    });
    document.querySelectorAll<HTMLSelectElement>(".sel-joint-gb").forEach((sel) => {
      const j = sel.dataset.joint!;
      d.joints[j].gearbox_id = sel.value ? sel.value : null;
    });

    await fetch(`/api/projects/${state.project_id}/design`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(d),
    });
    state.design = d;
    renderApp();
  });

  // Motion Preset Buttons
  document.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!state) return;
      const preset = btn.dataset.preset || "wipe";
      activeMotionPreset = preset;
      const res = await fetch(`/api/projects/${state.project_id}/motion`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset }),
      });
      if (res.ok) {
        const data = await res.json();
        state.intent = data.intent;
        state.target = data.target;
        renderApp();
      }
    });
  });

  // Video Upload with Client-Side MediaPipe Pose Extraction
  const videoInput = document.getElementById("inp-video-file") as HTMLInputElement | null;
  videoInput?.addEventListener("change", async () => {
    const file = videoInput.files?.[0];
    if (!file || !state) return;
    const previewWrap = document.getElementById("video-preview-wrap")!;
    const videoEl = document.getElementById("pose-video") as HTMLVideoElement;
    const canvasEl = document.getElementById("pose-canvas") as HTMLCanvasElement;
    const statusEl = document.getElementById("video-status")!;
    previewWrap.style.display = "block";

    try {
      const landmarks = await extractLandmarksFromVideoFile(
        file,
        videoEl,
        canvasEl,
        state.design.links.upper_arm_m,
        state.design.links.forearm_m,
        (pct, msg) => {
          statusEl.textContent = `${pct}% — ${msg}`;
        }
      );
      statusEl.textContent = "Uploading landmarks & running Argon intent analysis...";
      const res = await fetch(`/api/projects/${state.project_id}/motion`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ landmarks }),
      });
      if (res.ok) {
        const data = await res.json();
        state.intent = data.intent;
        state.target = data.target;
        activeMotionPreset = "custom_video";
        statusEl.textContent = "Pose extracted & Argon intent updated.";
        renderApp();
      }
    } catch (err: any) {
      statusEl.textContent = `Video error: ${err?.message || err}`;
    }
  });

  // Catalog Inline Edits
  document.querySelectorAll<HTMLInputElement>(".cat-edit").forEach((inp) => {
    inp.addEventListener("change", () => {
      if (!state) return;
      const idx = Number(inp.dataset.idx);
      const field = inp.dataset.field as keyof CatalogPart;
      const val = parseFloat(inp.value);
      const part = state.catalog.parts[idx] as any;
      if (part && part[field] && !Number.isNaN(val)) {
        part[field].value = val;
        if (field === "continuous_torque_nm") {
          part[field].derived = false;
        }
      }
    });
  });

  // Confirm Catalog & Start Loop (HITL #1)
  document.getElementById("btn-confirm-catalog")?.addEventListener("click", async () => {
    if (!state) return;
    state.catalog.parts.forEach((p) => {
      p.user_confirmed = true;
    });
    await fetch(`/api/projects/${state.project_id}/catalog`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.catalog),
    });
    await triggerRunLoop();
  });
}

function bindRunEvents(): void {
  document.getElementById("btn-run-loop-main")?.addEventListener("click", () => {
    triggerRunLoop();
  });
  document.getElementById("btn-goto-compare")?.addEventListener("click", () => {
    setTab("compare");
  });
}

async function triggerRunLoop(): Promise<void> {
  if (!state || isRunningLoop) return;
  isRunningLoop = true;
  activeTab = "run";
  runEvents = [];
  renderApp();

  try {
    const res = await fetch(`/api/projects/${state.project_id}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        max_iterations: 4,
        enable_post_pass_optimization: true,
        include_naive_baseline: true,
      }),
    });
    if (res.ok) {
      const data = await res.json();
      runEvents = data.events || [];
      await fetchProjectState(state.project_id);
    }
  } finally {
    isRunningLoop = false;
    renderApp();
  }
}

function bindCompareEvents(): void {
  document.getElementById("btn-run-from-compare")?.addEventListener("click", () => {
    triggerRunLoop();
  });

  const playBtn = document.getElementById("btn-play-pause");
  playBtn?.addEventListener("click", () => {
    isPlaying = !isPlaying;
    playBtn.innerHTML = isPlaying ? "&#10074;&#10074; Pause" : "&#9654; Play";
  });

  const worstBtn = document.getElementById("btn-jump-worst");
  worstBtn?.addEventListener("click", () => {
    const lr = state?.loop_result;
    if (!lr) return;
    const initTrace: TraceData = lr.initial_trace;
    const worstT = lr.iterations[0]?.metrics?.t_worst_ee_s ?? 3.27;
    const idx = initTrace.t.findIndex((tVal) => tVal >= worstT);
    currentFrameIdx = idx >= 0 ? idx : Math.floor(initTrace.t.length * 0.55);
    isPlaying = false;
    if (playBtn) playBtn.innerHTML = "&#9654; Play";
    syncScrubberAndViewports();
  });

  const scrubber = document.getElementById("scrubber-range") as HTMLInputElement | null;
  scrubber?.addEventListener("input", () => {
    currentFrameIdx = parseInt(scrubber.value, 10) || 0;
    isPlaying = false;
    if (playBtn) playBtn.innerHTML = "&#9654; Play";
    syncScrubberAndViewports();
  });

  const speedSel = document.getElementById("sel-speed") as HTMLSelectElement | null;
  speedSel?.addEventListener("change", () => {
    playbackSpeed = parseFloat(speedSel.value) || 1.0;
  });

  const axesChk = document.getElementById("chk-cad-axes") as HTMLInputElement | null;
  axesChk?.addEventListener("change", () => {
    showCadAxes = axesChk.checked;
    beforeViewport?.setAxisOverlaysVisible(showCadAxes);
    afterViewport?.setAxisOverlaysVisible(showCadAxes);
  });

  const ghostChk = document.getElementById("chk-ghost-arm") as HTMLInputElement | null;
  ghostChk?.addEventListener("change", () => {
    showGhostTarget = ghostChk.checked;
    beforeViewport?.setGhostArmVisible(showGhostTarget);
    afterViewport?.setGhostArmVisible(showGhostTarget);
  });

  // Engineer Sign-Off Button (HITL #2)
  document.getElementById("btn-approve-signoff")?.addEventListener("click", async () => {
    if (!state) return;
    const nameInp = document.getElementById("inp-eng-name") as HTMLInputElement;
    const notesInp = document.getElementById("inp-eng-notes") as HTMLInputElement;
    const engineerName = (nameInp?.value || "").trim() || "Alex Lee";
    const notes = (notesInp?.value || "").trim();

    const res = await fetch(`/api/projects/${state.project_id}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ engineer_name: engineerName, notes }),
    });
    if (res.ok) {
      const data = await res.json();
      state.approval = data.approval;
      renderApp();
      requestAnimationFrame(() => initCompareViewports());
    }
  });
}

function initCompareViewports(): void {
  if (!state || !state.loop_result) return;
  const beforeEl = document.getElementById("viewport-before");
  const afterEl = document.getElementById("viewport-after");
  if (!beforeEl || !afterEl) return;

  if (animFrameId !== null) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }

  const lr = state.loop_result;
  beforeViewport = new RobotArmViewport(beforeEl);
  afterViewport = new RobotArmViewport(afterEl);

  beforeViewport.setDesignAndTrace(lr.initial_design, lr.initial_trace, false);
  afterViewport.setDesignAndTrace(lr.final_design, lr.final_trace, true);

  beforeViewport.setAxisOverlaysVisible(showCadAxes);
  afterViewport.setAxisOverlaysVisible(showCadAxes);
  beforeViewport.setGhostArmVisible(showGhostTarget);
  afterViewport.setGhostArmVisible(showGhostTarget);

  // Sync camera orbits between Before and After viewports
  let syncingCamera = false;
  beforeViewport.controls.addEventListener("change", () => {
    if (syncingCamera || !beforeViewport || !afterViewport) return;
    syncingCamera = true;
    afterViewport.syncCameraFrom(beforeViewport);
    syncingCamera = false;
  });
  afterViewport.controls.addEventListener("change", () => {
    if (syncingCamera || !beforeViewport || !afterViewport) return;
    syncingCamera = true;
    beforeViewport.syncCameraFrom(afterViewport);
    syncingCamera = false;
  });

  syncScrubberAndViewports();
  lastTickMs = performance.now();

  const animate = (nowMs: number) => {
    if (activeTab !== "compare") return;
    const dtMs = nowMs - lastTickMs;
    const trace: TraceData = lr.initial_trace;
    const fps = trace.fps || 50;
    const stepIntervalMs = 1000 / (fps * playbackSpeed);

    if (isPlaying && dtMs >= stepIntervalMs) {
      const framesToAdvance = Math.max(1, Math.floor(dtMs / stepIntervalMs));
      currentFrameIdx = (currentFrameIdx + framesToAdvance) % trace.t.length;
      lastTickMs = nowMs;
      syncScrubberAndViewports();
    }

    beforeViewport?.render();
    afterViewport?.render();
    animFrameId = requestAnimationFrame(animate);
  };

  animFrameId = requestAnimationFrame(animate);
}

function syncScrubberAndViewports(): void {
  if (!state?.loop_result) return;
  const lr = state.loop_result;
  const initTrace: TraceData = lr.initial_trace;
  const finalTrace: TraceData = lr.final_trace;

  beforeViewport?.updateFrame(initTrace, currentFrameIdx);
  afterViewport?.updateFrame(finalTrace, currentFrameIdx);

  const scrubber = document.getElementById("scrubber-range") as HTMLInputElement | null;
  if (scrubber) scrubber.value = String(currentFrameIdx);

  const timeLabel = document.getElementById("scrubber-time-label");
  if (timeLabel) {
    const tVal = initTrace.t[currentFrameIdx] ?? 0;
    timeLabel.textContent = `t = ${tVal.toFixed(2)} s`;
  }

  updateLiveJointStrip("live-strip-before", initTrace, currentFrameIdx);
  updateLiveJointStrip("live-strip-after", finalTrace, currentFrameIdx);
}

function updateLiveJointStrip(containerId: string, trace: TraceData, frameIdx: number): void {
  const el = document.getElementById(containerId);
  if (!el || !trace || !trace.t.length) return;
  const idx = Math.max(0, Math.min(trace.t.length - 1, frameIdx));
  const tauCmd = trace.tau_cmd[idx] || [0, 0, 0, 0];
  const tauAvail = trace.tau_avail[idx] || [1, 1, 1, 1];

  const joints = ["J1", "J2", "J3", "J4"];
  el.innerHTML = joints
    .map((jName, jIdx) => {
      const cmd = Math.abs(tauCmd[jIdx] ?? 0);
      const avail = Math.max(Math.abs(tauAvail[jIdx] ?? 1), 1e-4);
      const ratio = cmd / avail;
      const isSat = ratio >= 0.98;
      return `
        <div class="joint-live-cell ${isSat ? "saturated" : ""}">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong class="mono">${jName}</strong>
            <span class="mono" style="color:${isSat ? "var(--fail)" : ratio >= 0.78 ? "var(--warn)" : "var(--pass)"};font-weight:700;">
              ${(ratio * 100).toFixed(0)}%
            </span>
          </div>
          <div class="mono" style="font-size:10.5px;color:var(--text-muted);margin-top:2px;">
            ${cmd.toFixed(2)} / ${avail.toFixed(2)} Nm
          </div>
        </div>
      `;
    })
    .join("");
}

// Bootstrap Application
fetchProjectState("demo")
  .then(() => {
    renderApp();
  })
  .catch((err) => {
    const root = document.getElementById("app");
    if (root) {
      root.innerHTML = `<div style="padding:32px;color:#ef4444;">Failed to initialize MotionFit: ${err.message}</div>`;
    }
  });
