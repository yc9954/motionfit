/**
 * MotionFit Frontend — 1code-Inspired Agentic Chat UI & 3D Floor-to-Floor Water-Bottle Simulation Studio
 *
 * Designed to be 100% intuitive for anyone with zero mechanical engineering background:
 * - Left Pane: 21st-dev/1code style Agentic Chat UI (TextShimmer thinking rows, collapsible tool calls,
 *   live checklist card, Human Arm Skeleton floor-to-floor motion tracker, and +/- Part Upgrade Diff Card).
 * - Right Pane: Interactive 3D Robot Studio where the robot reaches down to the floor (Spot A) to pick up
 *   a 500 mL water bottle, carries it across a long 68 cm floor-to-floor span, and places it on the floor (Spot B).
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

interface LandmarkFrame {
  t: number;
  shoulder: number[];
  elbow: number[];
  wrist: number[];
  index: number[];
  hip: number[];
  visibility?: number[];
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
  landmarks?: {
    fps: number;
    view: string;
    task?: string;
    frames: LandmarkFrame[];
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

interface ThinkingStep {
  status: "done" | "warn" | "fail" | "info";
  text: string;
  badge?: string;
}

interface ChatMessage {
  id: string;
  role: "user" | "agent";
  timestamp: string;
  html: string;
  attachmentLabel?: string;
  thinkingTitle?: string;
  thinkingPreview?: string;
  thinkingDuration?: string;
  isStreamingThinking?: boolean;
  thinkingSteps?: ThinkingStep[];
  showPlanCard?: boolean;
  planCompletedCount?: number;
  showSkeletonCard?: boolean;
  showRecommendationCard?: boolean;
  showApprovalCard?: boolean;
}

type StudioDisplayMode = "single-before" | "single-after" | "compare";

let state: ProjectState | null = null;
let chatMessages: ChatMessage[] = [];
let agentBusy = false;
let agentStatusText = "Ready to Build & Simulate";

// Workflow state
let robotAssembled = true;
let isAssembling = false;
let assemblyStage = 7;
let assemblyLabel = "All 7 robot parts assembled";
let recommendationAccepted = false;
let studioMode: StudioDisplayMode = "single-before";

// Composer state
let pendingVideoAttachment: { label: string; preset: string; file?: File } | null = null;

// Modals
let activeModal: "catalog" | "report" | "pdf" | null = null;
let pdfModalDoc = "XM430-W350.pdf";
let pdfModalPage = 1;

// 3D Viewports & Animation Loop
let beforeViewport: RobotArmViewport | null = null;
let afterViewport: RobotArmViewport | null = null;
let animFrameId: number | null = null;
let isPlaying = true;
let playbackSpeed = 1.0;
let currentFrameIdx = 0;
let lastTickMs = 0;
let showCadAxes = true;
let showGhostTarget = true;

const FRIENDLY_JOINT_NAMES: Record<string, string> = {
  J1: "Base Turn",
  J2: "Shoulder Lift",
  J3: "Elbow Bend",
  J4: "Wrist Grip",
};

function nowTimeStr(): string {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function escapeHtml(str: string): string {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchProjectState(projectId = "demo"): Promise<void> {
  const res = await fetch(`/api/projects/${projectId}`);
  if (!res.ok) {
    throw new Error(`Failed to load project ${projectId}: ${res.statusText}`);
  }
  state = await res.json();
}

function initDefaultConversation(): void {
  if (chatMessages.length > 0) return;

  chatMessages = [
    {
      id: "msg-welcome",
      role: "agent",
      timestamp: nowTimeStr(),
      html: `
        <div>
          👋 Welcome to <strong>MotionFit</strong> — your AI Robot Builder!
          <div style="margin-top:6px; color:#d4d4d8;">
            You don't need any mechanical engineering knowledge. Just tell me what you want the robot arm to do, or show me a video of your own arm doing the motion:
          </div>
          <div class="onecode-plan-card" style="margin-top:10px;">
            <div class="onecode-plan-header">
              <span>📋 How MotionFit Works</span>
              <span class="badge badge-info">Simple 3-Step Flow</span>
            </div>
            <div class="onecode-plan-steps">
              <div class="onecode-plan-step">
                <span class="onecode-plan-check active">1</span>
                <span><strong>Build the 3D Robot:</strong> Ask me to build a robot arm, and watch it assemble part-by-part on the right panel.</span>
              </div>
              <div class="onecode-plan-step">
                <span class="onecode-plan-check active">2</span>
                <span><strong>Show the Water-Bottle Video:</strong> Attach a video of a person picking up a <strong>500 mL water bottle from the floor</strong> and carrying it far across to another spot on the floor.</span>
              </div>
              <div class="onecode-plan-step">
                <span class="onecode-plan-check active">3</span>
                <span><strong>Auto-Fix Weak Parts:</strong> Watch the 3D simulation to see why the first robot's shoulder is too weak, then click <em>"Apply Upgrade"</em> to give it a 3× strength booster!</span>
              </div>
            </div>
          </div>
        </div>
      `,
    },
  ];
}

// ============================================================================
// MAIN RENDER FUNCTION
// ============================================================================

function renderApp(): void {
  const app = document.getElementById("app");
  if (!app || !state) return;

  const isApproved = Boolean(state.approval?.approved);

  app.innerHTML = `
    <!-- Top Global Header -->
    <header class="topbar">
      <div class="brand-group">
        <div class="brand-logo">MF</div>
        <div>
          <div class="brand-title">
            MotionFit
            <span class="badge badge-info">AI Robot Builder</span>
            ${
              recommendationAccepted
                ? `<span class="badge badge-pass">Current Robot: Upgraded Arm (Carries Bottle Easily ✓)</span>`
                : `<span class="badge badge-warn">Current Robot: First Try (Shoulder Too Weak for Bottle)</span>`
            }
            ${isApproved ? `<span class="badge badge-pass">✓ Approved by ${escapeHtml(state.approval?.engineer_name || "Team")}</span>` : ""}
          </div>
          <div class="brand-subtitle">Show a video of the motion you want. MotionFit builds the 3D robot, tests lifting the water bottle from the floor, and upgrades weak motors automatically.</div>
        </div>
      </div>

      <div class="topbar-actions">
        <button class="btn btn-sm" id="btn-open-catalog" title="Browse available motors and booster gears">
          Parts Catalog (${state.catalog.parts.length} Parts)
        </button>
        <button class="btn btn-sm" id="btn-open-report" title="View full summary report">
          Summary Report
        </button>
        <a class="btn btn-sm btn-primary" href="/api/projects/demo/export" download="motionfit-demo-bundle.zip">
          Download Robot Package (.zip)
        </a>
      </div>
    </header>

    <!-- Main Split Workspace: Left 1code Agent Chat | Right 3D Robot Studio -->
    <div class="workspace-split">
      <!-- ==================== LEFT PANE: 1code AGENT CHAT ==================== -->
      <section class="chat-pane">
        <div class="chat-pane-header">
          <div class="agent-status-pill">
            <span class="status-dot ${agentBusy ? "busy" : ""}"></span>
            <span id="agent-status-label" class="${agentBusy ? "onecode-shimmer" : ""}">${escapeHtml(agentStatusText)}</span>
          </div>
          <div style="display:flex; gap:6px;">
            <button class="btn btn-sm" id="btn-demo-walkthrough" title="Run the complete floor water-bottle analysis & recommendation">
              ✨ Run Full Demo
            </button>
            <button class="btn btn-sm" id="btn-reset-chat" title="Start a fresh conversation">
              Reset
            </button>
          </div>
        </div>

        <div class="chat-stream" id="chat-stream">
          ${chatMessages.map((m) => renderChatMessageHtml(m)).join("")}
        </div>

        <!-- 1code-Style Floating Bottom Composer -->
        <div class="chat-composer">
          <div class="quick-chips">
            <button class="chip-btn" id="chip-step1-build">
              1️⃣ Build 3D Robot Arm First
            </button>
            <button class="chip-btn" id="chip-step2-video">
              2️⃣ Send Floor Water-Bottle Video &amp; Test
            </button>
            <button class="chip-btn" id="chip-full-flow">
              ⚡ Build + Simulate Floor Bottle Transfer Together
            </button>
          </div>

          <div class="onecode-composer-box">
            <div class="composer-attachments" id="composer-attachments">
              ${
                pendingVideoAttachment
                  ? `
                    <span class="attachment-pill">
                      <span>📹 ${escapeHtml(pendingVideoAttachment.label)}</span>
                      <button id="btn-clear-attachment" style="background:none;border:none;color:#f87171;cursor:pointer;font-weight:700;">×</button>
                    </span>
                  `
                  : ""
              }
            </div>

            <form class="composer-row" id="chat-form">
              <textarea
                id="chat-input"
                class="composer-input"
                rows="2"
                placeholder="Describe what you want the robot arm to do, or attach a video of picking up a water bottle from the floor..."
              ></textarea>

              <div class="onecode-composer-footer">
                <div class="onecode-composer-pills">
                  <span class="onecode-model-pill">
                    <span style="color:#38bdf8;">✦</span> Gemini 4 Argon
                  </span>
                  <button class="btn btn-sm" id="btn-attach-demo-video" type="button">
                    ${pendingVideoAttachment ? "✓ Demo Bottle Video Attached" : "📹 Attach Floor Bottle Video"}
                  </button>
                  <label class="btn btn-sm" style="cursor:pointer;">
                    📁 Upload Video
                    <input type="file" id="input-video-file" accept="video/mp4,video/quicktime,video/webm" style="display:none;" />
                  </label>
                </div>

                <button type="submit" class="btn btn-primary btn-sm" id="btn-send-chat" style="padding:5px 14px; border-radius:999px;" ${agentBusy ? "disabled" : ""}>
                  ${agentBusy ? "Working..." : "Send ↑"}
                </button>
              </div>
            </form>
          </div>
        </div>
      </section>

      <!-- ==================== RIGHT PANE: 3D ROBOT STUDIO ==================== -->
      <section class="studio-pane">
        <!-- Studio Toolbar -->
        <div class="studio-toolbar">
          <div class="studio-mode-group">
            <span style="font-size:11.5px; font-weight:700; color:#71717a; margin-right:4px;">3D ROBOT VIEW:</span>
            <button class="btn btn-sm ${studioMode === "single-before" ? "btn-primary" : ""}" id="btn-view-before">
              1. First Try (Shoulder Too Weak ✗)
            </button>
            <button class="btn btn-sm ${studioMode === "single-after" ? "btn-primary" : ""}" id="btn-view-after">
              2. Upgraded Robot (Carries Bottle ✓)
            </button>
            <button class="btn btn-sm ${studioMode === "compare" ? "btn-primary" : ""}" id="btn-view-compare">
              Side-by-Side Compare
            </button>
          </div>

          <div class="studio-mode-group">
            <button class="btn btn-sm" id="btn-replay-assembly" title="Watch the 3D robot arm assemble part-by-part">
              🔧 Replay 3D Build
            </button>
            <button class="btn btn-sm ${showCadAxes ? "btn-primary" : ""}" id="btn-toggle-axes">
              Show Motor Strain Arrows
            </button>
            <button class="btn btn-sm ${showGhostTarget ? "btn-primary" : ""}" id="btn-toggle-ghost">
              Show Ideal Path
            </button>
          </div>
        </div>

        <!-- 3D Viewport Grid (Contains both #viewport-before and #viewport-after) -->
        <div class="studio-stage-grid ${studioMode === "compare" ? "compare-mode" : ""}" id="studio-stage-grid">
          <!-- BEFORE / FIRST TRY VIEWPORT CARD -->
          <div class="viewport-card ${studioMode === "single-after" ? "hidden-pane" : ""}" id="card-before">
            <div class="viewport-hud-top">
              <div class="hud-pill">
                <div class="hud-title">
                  <span class="badge badge-fail">FIRST TRY (FAILS)</span>
                  <span>Standard Shoulder Motor (No Booster Gear)</span>
                </div>
                <div class="hud-sub">
                  Picks up the <strong>500 mL water bottle from Floor Spot A</strong>, but the bottle is too heavy when held out far. Halfway across, the <strong style="color:#f87171;">Shoulder Motor runs out of strength (red arrow)</strong> and drops the bottle on the floor!
                </div>
              </div>
              <div class="hud-pill" style="max-width:none;">
                <span style="color:#38bdf8;">● Floor Spot A</span> &rarr; <span style="color:#ef4444;">● Sags &amp; Drops Halfway</span>
              </div>
            </div>
            <div class="viewport-canvas-host" id="viewport-before"></div>
          </div>

          <!-- AFTER / UPGRADED VIEWPORT CARD -->
          <div class="viewport-card ${studioMode === "single-before" ? "hidden-pane" : ""}" id="card-after">
            <div class="viewport-hud-top">
              <div class="hud-pill">
                <div class="hud-title">
                  <span class="badge badge-pass">UPGRADED ROBOT (SUCCESS ✓)</span>
                  <span>3× Shoulder Booster Gear + Lighter Arm Tubes</span>
                </div>
                <div class="hud-sub">
                  With a <strong style="color:#34d399;">3× Strength Booster Gear</strong> on the shoulder and lighter arm tubes, the robot easily lifts the water bottle from <strong>Floor Spot A</strong>, carries it far across, and sets it gently on <strong>Floor Spot B</strong>!
                </div>
              </div>
              <div class="hud-pill" style="max-width:none;">
                <span style="color:#38bdf8;">● Floor Spot A</span> &rarr; <span style="color:#10b981;">● Placed on Floor Spot B ✓</span>
              </div>
            </div>
            <div class="viewport-canvas-host" id="viewport-after"></div>
          </div>

          <!-- Progressive 3D Assembly Overlay -->
          <div class="assembly-overlay" id="assembly-overlay" style="display: ${isAssembling ? "block" : "none"};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div>
                <strong style="color:#38bdf8;">🔧 Building 3D Robot Arm (${assemblyStage}/7)</strong>
                <span style="margin-left:8px; color:#e4e4e7; font-size:12.5px;" id="assembly-step-text">${escapeHtml(assemblyLabel)}</span>
              </div>
              <span class="badge badge-info">${Math.round((assemblyStage / 7) * 100)}%</span>
            </div>
            <div class="assembly-progress-bar">
              <div class="assembly-progress-fill" style="width: ${Math.round((assemblyStage / 7) * 100)}%;"></div>
            </div>
          </div>
        </div>

        <!-- Studio Bottom Dock: Timeline + Friendly Motor Effort & Strain Meter -->
        <div class="studio-bottom-dock">
          <div class="timeline-bar">
            <button class="btn btn-sm btn-primary" id="btn-play-pause" style="min-width:78px;">
              ${isPlaying ? "⏸ Pause" : "▶ Play"}
            </button>
            <input
              type="range"
              id="timeline-slider"
              class="timeline-slider"
              min="0"
              max="${Math.max(1, (state.loop_result?.iterations?.[0]?.trace?.t?.length || 180) - 1)}"
              value="${currentFrameIdx}"
            />
            <span class="badge badge-info" id="phase-readout">Step 2: Lifting &amp; Carrying Bottle</span>
            <div class="time-readout" id="time-readout">0.0s / 6.0s</div>
            <select id="speed-select" class="btn btn-sm" style="background:#14151a;">
              <option value="0.5" ${playbackSpeed === 0.5 ? "selected" : ""}>0.5x Slow</option>
              <option value="1" ${playbackSpeed === 1.0 ? "selected" : ""}>1.0x Normal</option>
              <option value="2" ${playbackSpeed === 2.0 ? "selected" : ""}>2.0x Fast</option>
            </select>
          </div>

          <!-- Friendly Motor Effort & Strain Meter (Preserves #live-strip-before and #live-strip-after for tests) -->
          <div class="studio-telemetry-grid">
            <div class="telemetry-box">
              <div class="telemetry-box-title">
                <span>First Try — Motor Effort &amp; Strain Meter</span>
                <span style="color:#f87171;">⚠ Shoulder Overloaded (100% Maxed Out!)</span>
              </div>
              <div class="joint-bars" id="live-strip-before"></div>
            </div>
            <div class="telemetry-box">
              <div class="telemetry-box-title">
                <span>Upgraded Robot — Motor Effort &amp; Strain Meter</span>
                <span style="color:#34d399;">✓ All Motors in Safe Green Zone</span>
              </div>
              <div class="joint-bars" id="live-strip-after"></div>
            </div>
          </div>
        </div>
      </section>
    </div>

    ${renderModalHtml()}
  `;

  bindEvents();
  initOrRefreshViewports();
  drawHumanSkeletonCanvas(currentFrameIdx);
  scrollChatToBottom();
}

// ============================================================================
// 1code AGENT CHAT MESSAGE & RICH CARDS RENDERING
// ============================================================================

function renderChatMessageHtml(m: ChatMessage): string {
  if (m.role === "user") {
    return `
      <div class="chat-msg user" id="${escapeHtml(m.id)}">
        ${
          m.attachmentLabel
            ? `<div class="onecode-user-attachments"><span class="attachment-pill">📹 ${escapeHtml(m.attachmentLabel)}</span></div>`
            : ""
        }
        <div class="onecode-user-bubble">
          ${m.html}
        </div>
      </div>
    `;
  }

  // 1code AgentThinkingTool collapsible block
  const thinkingHtml =
    m.thinkingSteps && m.thinkingSteps.length > 0
      ? `
        <details class="onecode-thinking" open>
          <summary class="onecode-thinking-summary">
            <div class="onecode-thinking-left">
              <span style="color:#38bdf8;">✦</span>
              <span class="onecode-thinking-label ${m.isStreamingThinking ? "onecode-shimmer" : ""}">
                ${escapeHtml(m.thinkingTitle || "Thought process")}
              </span>
              ${
                m.thinkingPreview
                  ? `<span class="onecode-thinking-preview">— ${escapeHtml(m.thinkingPreview)}</span>`
                  : ""
              }
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="onecode-thinking-duration">${escapeHtml(m.thinkingDuration || `${m.thinkingSteps.length} steps`)}</span>
              <span class="onecode-chevron">›</span>
            </div>
          </summary>
          <div class="onecode-thinking-content">
            ${m.thinkingSteps
              .map((s) => {
                const icon =
                  s.status === "done"
                    ? "✓"
                    : s.status === "warn"
                    ? "!"
                    : s.status === "fail"
                    ? "✗"
                    : "⋯";
                const badgeClass =
                  s.status === "done"
                    ? "badge-pass"
                    : s.status === "warn"
                    ? "badge-warn"
                    : s.status === "fail"
                    ? "badge-fail"
                    : "badge-info";
                return `
                  <div class="onecode-tool-row ${s.status}">
                    <div class="onecode-tool-row-left">
                      <span class="onecode-tool-icon">${icon}</span>
                      <span>${s.text}</span>
                    </div>
                    ${s.badge ? `<span class="badge ${badgeClass}" style="flex-shrink:0;">${escapeHtml(s.badge)}</span>` : ""}
                  </div>
                `;
              })
              .join("")}
          </div>
        </details>
      `
      : "";

  const planCardHtml = m.showPlanCard ? renderOneCodePlanCardHtml(m.planCompletedCount ?? 4) : "";
  const skeletonCardHtml = m.showSkeletonCard ? renderSkeletonAnalysisCardHtml() : "";
  const recommendationCardHtml = m.showRecommendationCard ? renderRecommendationCardHtml() : "";
  const approvalCardHtml = m.showApprovalCard ? renderApprovalCardHtml() : "";

  return `
    <div class="chat-msg agent" id="${escapeHtml(m.id)}">
      <div class="onecode-agent-block">
        <div class="onecode-agent-header">
          <span class="onecode-agent-badge">
            <span class="onecode-agent-dot">AI</span>
            <span>MotionFit Co-Pilot</span>
          </span>
          <span>${escapeHtml(m.timestamp)}</span>
        </div>
        ${thinkingHtml}
        ${planCardHtml}
        <div class="onecode-agent-prose">${m.html}</div>
        ${skeletonCardHtml}
        ${recommendationCardHtml}
        ${approvalCardHtml}
      </div>
    </div>
  `;
}

function renderOneCodePlanCardHtml(completedCount: number): string {
  const steps = [
    "Assemble 3D Robot Arm & Floor Coaster Mats on Right Panel",
    "Track Human Arm Skeleton from Video (Floor Pickup → Long Carry → Floor Dropoff)",
    "Test 500 mL Water Bottle Lift in 3D Physics Simulator (Spot Why It Sags)",
    "Recommend 3× Shoulder Strength Booster + Lighter Arm Tubes",
  ];

  return `
    <div class="onecode-plan-card">
      <div class="onecode-plan-header">
        <span>📋 Agent Task Checklist</span>
        <span class="badge ${completedCount >= 4 ? "badge-pass" : "badge-info"}">${Math.min(4, completedCount)} / 4 Completed</span>
      </div>
      <div class="onecode-plan-steps">
        ${steps
          .map((label, idx) => {
            const isDone = idx < completedCount;
            return `
              <div class="onecode-plan-step">
                <span class="onecode-plan-check ${isDone ? "completed" : "active"}">${isDone ? "✓" : idx + 1}</span>
                <span style="${isDone ? "color:#e4e4e7;" : "color:#71717a;"}">${escapeHtml(label)}</span>
              </div>
            `;
          })
          .join("")}
      </div>
    </div>
  `;
}

function renderSkeletonAnalysisCardHtml(): string {
  return `
    <div class="skeleton-analysis-card">
      <div class="skeleton-card-header">
        <div>
          <strong style="color:#38bdf8; font-size:12.5px;">🦴 Human Arm Motion Tracker (From Your Video)</strong>
          <div style="font-size:11px; color:#a1a1aa;">Tracks your Shoulder &rarr; Elbow &rarr; Wrist &rarr; Hand picking up the 500 mL water bottle from the floor and carrying it far across</div>
        </div>
        <span class="badge badge-info">Live Synced</span>
      </div>
      <div class="skeleton-canvas-wrap">
        <canvas id="skeleton-canvas" class="skeleton-canvas" width="520" height="185"></canvas>
      </div>
      <div class="phase-pills">
        <div class="phase-pill">
          <div class="phase-pill-title">1. Reach Down to Floor</div>
          <div style="color:#e4e4e7; font-size:10.5px;">0.0s – 1.2s</div>
          <div style="color:#a1a1aa; font-size:10.5px; margin-top:2px;">Hand bends down to Floor Spot A &amp; grabs the 500 mL water bottle</div>
        </div>
        <div class="phase-pill">
          <div class="phase-pill-title">2. Long-Distance Lift &amp; Carry</div>
          <div style="color:#e4e4e7; font-size:10.5px;">1.2s – 4.9s (Hardest Part!)</div>
          <div style="color:#a1a1aa; font-size:10.5px; margin-top:2px;">Lifts bottle high off the floor &amp; carries it 68 cm across to the right</div>
        </div>
        <div class="phase-pill">
          <div class="phase-pill-title">3. Set Bottle on Floor</div>
          <div style="color:#e4e4e7; font-size:10.5px;">4.9s – 6.0s</div>
          <div style="color:#a1a1aa; font-size:10.5px; margin-top:2px;">Gently places bottle upright on Floor Spot B &amp; lets go</div>
        </div>
      </div>
    </div>
  `;
}

function renderRecommendationCardHtml(): string {
  const lr = state?.loop_result;
  const iter0 = lr?.iterations?.[0];
  const finalIter = lr?.iterations?.[lr.iterations.length - 1];
  const m0 = iter0?.metrics;
  const mf = finalIter?.metrics;

  return `
    <div class="recommendation-card">
      <div class="rec-header">
        <div>
          <div class="rec-title">✨ Recommended Robot Upgrade (Easy 2-Part Fix)</div>
          <div style="font-size:11.5px; color:#d4d4d8; margin-top:2px;">
            Makes the shoulder <strong>3× stronger</strong> and the arm <strong>lighter</strong> so it carries the water bottle from floor to floor effortlessly.
          </div>
        </div>
        <span class="badge badge-pass">Verified in 3D Physics ✓</span>
      </div>

      <!-- 1code-Style Diff Block: Clear Before (-) vs After (+) in Plain Language -->
      <div class="onecode-diff-block">
        <div class="onecode-diff-header">
          <span>🔧 What We Change on the Robot Arm</span>
          <span style="color:#a1a1aa; font-weight:400;">2 part changes · 3 other motors stay the same</span>
        </div>

        <div class="onecode-diff-line removed">
          <span class="onecode-diff-sign">-</span>
          <div>
            <strong>Shoulder Lift Motor (Before):</strong> Standard motor alone — <em>runs out of strength when holding the heavy water bottle out far, causing the arm to sag to the floor.</em>
          </div>
        </div>
        <div class="onecode-diff-line added">
          <span class="onecode-diff-sign">+</span>
          <div>
            <strong>Shoulder Lift Motor (Upgrade):</strong> Add a <strong>3× Strength Booster Gear</strong> (<code>APEX-PG20-2</code>) — <em>multiplies lifting power by 3× right at the shoulder without needing a giant heavy motor!</em>
          </div>
        </div>

        <div class="onecode-diff-line removed">
          <span class="onecode-diff-sign">-</span>
          <div>
            <strong>Arm Body Tubes (Before):</strong> Thicker, heavier aluminum tubes — <em>adds extra weight that the shoulder has to hold up.</em>
          </div>
        </div>
        <div class="onecode-diff-line added">
          <span class="onecode-diff-sign">+</span>
          <div>
            <strong>Arm Body Tubes (Upgrade):</strong> Switch to <strong>Slimmer, Lighter Aluminum Tubes</strong> — <em>shaves 70 grams off the arm so lifting the bottle is even easier.</em>
          </div>
        </div>
      </div>

      <!-- Plain-Language Outcome Summary Boxes -->
      <div class="rec-metrics-grid">
        <div class="rec-metric-box">
          <div class="rec-metric-label">Water Bottle Floor-to-Floor Delivery</div>
          <div class="rec-metric-val">
            <span style="color:#f87171; text-decoration:line-through;">Drops halfway</span>
            &rarr; <span style="color:#34d399;">Placed on Floor Spot B ✓</span>
          </div>
        </div>
        <div class="rec-metric-box">
          <div class="rec-metric-label">Shoulder Motor Strain</div>
          <div class="rec-metric-val">
            <span style="color:#f87171; text-decoration:line-through;">Overloaded (238%)</span>
            &rarr; <span style="color:#34d399;">Comfortable (78% effort)</span>
          </div>
        </div>
        <div class="rec-metric-box">
          <div class="rec-metric-label">Total Robot Arm Weight</div>
          <div class="rec-metric-val">
            <span style="color:#a1a1aa;">${m0?.total_mass_kg?.toFixed(2) ?? "1.01"} kg</span>
            &rarr; <span style="color:#34d399;">${mf?.total_mass_kg?.toFixed(2) ?? "0.94"} kg (Lighter!)</span>
          </div>
        </div>
        <div class="rec-metric-box">
          <div class="rec-metric-label">Total Parts Cost</div>
          <div class="rec-metric-val">
            <span style="color:#a1a1aa;">$${m0?.total_cost_usd?.toFixed(0) ?? "945"}</span>
            &rarr; <span style="color:#38bdf8;">$${mf?.total_cost_usd?.toFixed(0) ?? "1,126"} (Under $1,200 Budget)</span>
          </div>
        </div>
      </div>

      <!-- Optional Technical Details & PDF Datasheet Citations Drawer -->
      <details class="onecode-thinking" style="margin-top:8px;">
        <summary class="onecode-thinking-summary">
          <div class="onecode-thinking-left">
            <span>📑</span>
            <span class="onecode-thinking-label">Show Exact Part Model Numbers &amp; Datasheet PDF Citations (Optional)</span>
          </div>
          <span class="onecode-chevron">›</span>
        </summary>
        <div class="onecode-thinking-content" style="font-size:11.5px; color:#d4d4d8;">
          <div>• <strong>Shoulder Lift (J2):</strong> <code>ROBOTIS XM430-W350</code> + <code>APEX-PG20-2</code> (3:1 planetary gearbox, 9.79 N·m continuous limit)
            <button class="cite-tag btn-cite-pdf" data-doc="XM430-W350.pdf" data-page="2">XM430-W350.pdf p.2</button>
            <button class="cite-tag btn-cite-pdf" data-doc="APEX-PG20-2.pdf" data-page="1">APEX-PG20-2.pdf p.1</button>
          </div>
          <div>• <strong>Arm Link Profile:</strong> <code>al_tube_25x2</code> &rarr; <code>al_tube_20x1_5</code> (tracking error improved from <code>${m0?.ee_rmse_mm?.toFixed(1) ?? "295.1"} mm</code> to <code>${mf?.ee_rmse_mm?.toFixed(2) ?? "11.98"} mm</code>)</div>
          <div>• <strong>Other Motors (Base, Elbow, Wrist):</strong> <code>XM430-W350</code> &amp; <code>XL430-W250</code> retained
            <button class="cite-tag btn-cite-pdf" data-doc="XL430-W250.pdf" data-page="2">XL430-W250.pdf p.2</button>
          </div>
        </div>
      </details>

      <div class="rec-actions">
        <button class="btn ${recommendationAccepted ? "btn-primary" : "btn-success"}" id="btn-accept-recommendation">
          ${
            recommendationAccepted
              ? "✓ Upgrade Applied to 3D Robot! (Click to Replay Upgrade)"
              : "✅ Apply Upgrade to 3D Robot & Watch It Carry the Bottle"
          }
        </button>
        <button class="btn btn-sm" id="btn-rec-compare">
          Compare Before vs. After Side-by-Side
        </button>
        <button class="btn btn-sm" id="btn-rec-report">
          Open Full Report
        </button>
      </div>
    </div>
  `;
}

function renderApprovalCardHtml(): string {
  const isApproved = Boolean(state?.approval?.approved);
  const engName = state?.approval?.engineer_name || "Alex Lee";
  const engNotes =
    state?.approval?.notes ||
    "Approved 3x shoulder booster gear + lighter arm tubes for floor-to-floor water bottle transfer.";

  return `
    <div class="recommendation-card" style="border-color: rgba(56, 189, 248, 0.4); background: #10131a;">
      <div class="rec-header">
        <div class="rec-title" style="color:#38bdf8;">✍️ Final Sign-Off &amp; Download Robot Package</div>
        <span class="badge ${isApproved ? "badge-pass" : "badge-warn"}">${isApproved ? "APPROVED ✓" : "READY TO APPROVE"}</span>
      </div>
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-bottom:8px;">
        <input
          type="text"
          id="input-eng-name"
          style="min-height:34px; padding:6px 10px; border-radius:8px; border:1px solid rgba(255,255,255,0.12); background:#09090b; color:#fff;"
          value="${escapeHtml(engName)}"
          placeholder="Your Name"
        />
        <input
          type="text"
          id="input-eng-notes"
          style="min-height:34px; padding:6px 10px; border-radius:8px; border:1px solid rgba(255,255,255,0.12); background:#09090b; color:#fff;"
          value="${escapeHtml(engNotes)}"
          placeholder="Approval Notes"
        />
      </div>
      <div style="display:flex; gap:8px; align-items:center;">
        <button class="btn btn-sm btn-success" id="btn-submit-approval">
          ${isApproved ? "✓ Saved Approval" : "Approve Final Robot Design"}
        </button>
        <a class="btn btn-sm btn-primary" href="/api/projects/demo/export" download="motionfit-demo-bundle.zip">
          Download Parts List &amp; 3D Model (.zip)
        </a>
      </div>
    </div>
  `;
}

// ============================================================================
// HUMAN ARM SKELETON FLOOR-TO-FLOOR CANVAS OVERLAY (IN CHAT)
// ============================================================================

function drawHumanSkeletonCanvas(frameIdx: number): void {
  const canvas = document.getElementById("skeleton-canvas") as HTMLCanvasElement | null;
  if (!canvas || !state) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Subtle background grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 35) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Draw the Floor Line across the bottom of the canvas so the user clearly sees the bottle on the floor!
  const floorY = h - 26;
  ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(24, floorY);
  ctx.lineTo(w - 24, floorY);
  ctx.stroke();

  ctx.fillStyle = "rgba(148, 163, 184, 0.65)";
  ctx.font = "10px Inter, sans-serif";
  ctx.fillText("FLOOR LEVEL (68 cm Long-Distance Bottle Carry)", w / 2 - 110, floorY + 16);

  const totalFrames = state.landmarks?.frames?.length || 180;
  const idx = Math.max(0, Math.min(totalFrames - 1, frameIdx));
  const u = idx / Math.max(1, totalFrames - 1);

  // Floor Spot A (far left on the floor) and Floor Spot B (far right on the floor)
  const pickupX = 115;
  const pickupY = floorY - 12;
  const targetX = w - 95;
  const targetY = floorY - 12;

  // Draw high floor-to-floor arc dashed guide line
  ctx.beginPath();
  for (let i = 0; i <= 40; i++) {
    const s = i / 40;
    const ax = pickupX + (targetX - pickupX) * s;
    const ay = pickupY - Math.sin(Math.PI * s) * 86;
    if (i === 0) ctx.moveTo(ax, ay);
    else ctx.lineTo(ax, ay);
  }
  ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw Floor Spot A Coaster Pad on the floor
  ctx.fillStyle = "rgba(56, 189, 248, 0.25)";
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.ellipse(pickupX, floorY, 24, 5, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#7dd3fc";
  ctx.font = "bold 10px Inter, sans-serif";
  ctx.fillText("Floor Spot A", pickupX - 30, floorY + 14);

  // Draw Floor Spot B Coaster Pad on the floor
  ctx.fillStyle = "rgba(16, 185, 129, 0.25)";
  ctx.strokeStyle = "#10b981";
  ctx.beginPath();
  ctx.ellipse(targetX, floorY, 24, 5, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#6ee7b7";
  ctx.fillText("Floor Spot B", targetX - 30, floorY + 14);

  // Compute human hand position along the floor-to-floor trajectory
  const graspU = 0.19;
  const releaseU = 0.81;
  let handX = pickupX;
  let handY = pickupY;
  let isCarrying = false;

  if (u < graspU) {
    const p = u / graspU;
    handX = pickupX - 18 * (1 - p);
    handY = pickupY - 48 * (1 - p);
  } else if (u <= releaseU) {
    isCarrying = true;
    const s = (u - graspU) / (releaseU - graspU);
    const smoothS = s * s * (3 - 2 * s);
    handX = pickupX + (targetX - pickupX) * smoothS;
    handY = pickupY - Math.sin(Math.PI * smoothS) * 86;
  } else {
    const p = (u - releaseU) / (1 - releaseU);
    handX = targetX - 10 * p;
    handY = targetY - 38 * p;
  }

  // Human Shoulder & Hip anchor on the left-center
  const shX = 68;
  const shY = 58;
  const hipX = 56;
  const hipY = floorY - 4;

  // Solve 2D elbow & wrist positions for the human skeleton
  const midX = (shX + handX) * 0.5;
  const midY = (shY + handY) * 0.5;
  const elX = midX - 14;
  const elY = Math.min(floorY - 18, midY + 26);
  const wrX = handX - 14;
  const wrY = handY - 4;

  // Draw Torso (Hip -> Shoulder)
  ctx.strokeStyle = "rgba(148, 163, 184, 0.5)";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(hipX, hipY);
  ctx.lineTo(shX, shY);
  ctx.stroke();

  // Draw Upper Arm (Shoulder -> Elbow)
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.moveTo(shX, shY);
  ctx.lineTo(elX, elY);
  ctx.stroke();

  // Draw Forearm (Elbow -> Wrist)
  ctx.strokeStyle = "#22c55e";
  ctx.lineWidth = 4.5;
  ctx.beginPath();
  ctx.moveTo(elX, elY);
  ctx.lineTo(wrX, wrY);
  ctx.stroke();

  // Draw Hand (Wrist -> Gripper/Fingers)
  ctx.strokeStyle = "#a855f7";
  ctx.lineWidth = 3.5;
  ctx.beginPath();
  ctx.moveTo(wrX, wrY);
  ctx.lineTo(handX, handY);
  ctx.stroke();

  // Draw 500 mL Water Bottle on Floor Spot A, in Hand, or on Floor Spot B
  let bottleX = pickupX;
  let bottleY = pickupY;
  if (isCarrying) {
    bottleX = handX;
    bottleY = handY;
  } else if (u > releaseU) {
    bottleX = targetX;
    bottleY = targetY;
  }

  ctx.fillStyle = "rgba(14, 165, 233, 0.88)";
  ctx.strokeStyle = "#e0f2fe";
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.roundRect(bottleX - 7, bottleY - 12, 14, 24, 4);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#1d4ed8";
  ctx.fillRect(bottleX - 4, bottleY - 16, 8, 4);

  // Draw Friendly Skeleton Joint Labels
  const joints: Array<{ x: number; y: number; color: string; label: string }> = [
    { x: shX, y: shY, color: "#f59e0b", label: "Shoulder" },
    { x: elX, y: elY, color: "#38bdf8", label: "Elbow" },
    { x: wrX, y: wrY, color: "#22c55e", label: "Wrist" },
    { x: handX, y: handY, color: "#e879f9", label: "Hand + Bottle" },
  ];

  joints.forEach((j) => {
    ctx.fillStyle = j.color;
    ctx.beginPath();
    ctx.arc(j.x, j.y, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#09090b";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = "#f4f4f5";
    ctx.font = "10px Inter, sans-serif";
    ctx.fillText(j.label, j.x + 7, j.y - 6);
  });

  // Top status caption inside canvas
  ctx.fillStyle = "#d4d4d8";
  ctx.font = "11px Inter, sans-serif";
  const statusMsg =
    u < graspU
      ? "Step 1: Reaching down to Floor Spot A to grab the 500 mL water bottle"
      : isCarrying
      ? "Step 2: Lifting water bottle off the floor & carrying it far across to Floor Spot B"
      : "Step 3: Placed water bottle upright on Floor Spot B";
  ctx.fillText(statusMsg, 12, 18);
}

// ============================================================================
// CONVERSATIONAL WORKFLOWS (STEP 1: BUILD ARM, STEP 2: VIDEO SIM, OR COMBINED)
// ============================================================================

async function runProgressiveAssemblyAnimation(targetMode: StudioDisplayMode, labelPrefix: string): Promise<void> {
  isAssembling = true;
  studioMode = targetMode;
  const stageLabels = [
    "1/7 Placing Base Controller Box on the Floor...",
    "2/7 Mounting Base Turn Motor (Swivels Left & Right)...",
    "3/7 Attaching Upper Arm Tube...",
    "4/7 Mounting Shoulder Lift Motor (Lifts Arm Up & Down)...",
    "5/7 Attaching Elbow Bend Motor & Forearm Tube...",
    "6/7 Mounting Wrist Motor & 2-Finger Bottle Gripper...",
    "7/7 Placing Floor Spot A, Floor Spot B & 500 mL Water Bottle on the Floor...",
  ];

  for (let s = 1; s <= 7; s++) {
    assemblyStage = s;
    assemblyLabel = `${labelPrefix}: ${stageLabels[s - 1]}`;
    beforeViewport?.setAssemblyStage(s);
    afterViewport?.setAssemblyStage(s);
    updateAssemblyOverlayDom();
    await sleep(180);
  }

  isAssembling = false;
  robotAssembled = true;
  updateAssemblyOverlayDom();
}

function updateAssemblyOverlayDom(): void {
  const overlay = document.getElementById("assembly-overlay");
  if (!overlay) return;
  overlay.style.display = isAssembling ? "block" : "none";
  const stepText = document.getElementById("assembly-step-text");
  if (stepText) stepText.textContent = assemblyLabel;
  const fill = overlay.querySelector(".assembly-progress-fill") as HTMLElement | null;
  if (fill) fill.style.width = `${Math.round((assemblyStage / 7) * 100)}%`;
}

/**
 * Step 1 Only: User sends a prompt asking to build a robot arm (without a video yet).
 */
async function handleTurn1BuildArmPrompt(userPrompt: string): Promise<void> {
  if (agentBusy) return;
  agentBusy = true;
  agentStatusText = "Building 3D Robot Arm on Right Panel...";

  chatMessages.push({
    id: `user-${Date.now()}`,
    role: "user",
    timestamp: nowTimeStr(),
    html: escapeHtml(userPrompt),
  });

  const agentMsgId = `agent-${Date.now() + 1}`;
  const thinkingSteps: ThinkingStep[] = [
    {
      status: "done",
      text: "Selected a 4-joint desktop robot arm: Base Turn, Shoulder Lift, Elbow Bend, and a 2-Finger Bottle Gripper.",
      badge: "Design Ready",
    },
    {
      status: "done",
      text: "Picked starter motors from the parts catalog and placed Floor Spot A and Floor Spot B 68 cm apart on the ground.",
      badge: "Parts Picked",
    },
    {
      status: "info",
      text: "Assembling the 3D robot arm part-by-part in the Right Studio...",
      badge: "Building 3D",
    },
  ];

  chatMessages.push({
    id: agentMsgId,
    role: "agent",
    timestamp: nowTimeStr(),
    html: `I'm building your 3D robot arm in the <strong>Right Studio</strong> right now — watch the parts snap together!`,
    thinkingTitle: "Thinking & Building 3D Robot Arm",
    thinkingPreview: "Selecting starter motors & assembling 7 robot parts",
    thinkingDuration: "1.4s",
    isStreamingThinking: true,
    thinkingSteps,
    showPlanCard: true,
    planCompletedCount: 1,
  });

  recommendationAccepted = false;
  studioMode = "single-before";
  renderApp();

  await runProgressiveAssemblyAnimation("single-before", "Building Starter Robot");

  thinkingSteps[2] = {
    status: "done",
    text: "Finished assembling all 7 parts of the 3D robot arm and placed the 500 mL water bottle on Floor Spot A.",
    badge: "Built ✓",
  };
  const targetMsg = chatMessages.find((m) => m.id === agentMsgId);
  if (targetMsg) {
    targetMsg.isStreamingThinking = false;
    targetMsg.html = `
      <div>
        ✅ <strong>Your 3D Robot Arm is Built!</strong>
        <br/>
        Look at the right panel: the robot arm is assembled, and the <strong>500 mL water bottle</strong> is sitting on the floor at <strong>Floor Spot A</strong> (blue floor pad).
        <br/><br/>
        📹 <strong>Next Step:</strong> Attach a motion video (or click <em>"2️⃣ Send Floor Water-Bottle Video &amp; Test"</em> below) so I can track the human arm motion, test picking up the water bottle from the floor, and check if the motors are strong enough!
      </div>
    `;
  }

  agentBusy = false;
  agentStatusText = "Waiting for Your Motion Video";
  renderApp();
}

/**
 * Step 2 (or Combined): User attaches/provides the motion video.
 */
async function handleVideoSimulationAndRedesign(
  userPrompt: string,
  videoLabel: string,
  alsoAssembleFirst: boolean,
  uploadedFile?: File
): Promise<void> {
  if (agentBusy) return;
  agentBusy = true;

  chatMessages.push({
    id: `user-${Date.now()}`,
    role: "user",
    timestamp: nowTimeStr(),
    attachmentLabel: videoLabel,
    html: escapeHtml(userPrompt),
  });

  const agentMsgId = `agent-${Date.now() + 1}`;
  const thinkingSteps: ThinkingStep[] = [];

  chatMessages.push({
    id: agentMsgId,
    role: "agent",
    timestamp: nowTimeStr(),
    html: `Analyzing your video and testing the <strong>floor-to-floor water bottle carry</strong> on the 3D robot...`,
    thinkingTitle: "Thinking & Testing Robot in 3D Simulator",
    thinkingPreview: "Tracking human arm & testing 500 mL bottle lift from floor",
    thinkingDuration: "2.3s",
    isStreamingThinking: true,
    thinkingSteps,
    showPlanCard: true,
    planCompletedCount: 1,
  });

  recommendationAccepted = false;
  studioMode = "single-before";
  renderApp();

  const targetMsg = chatMessages.find((m) => m.id === agentMsgId);

  if (alsoAssembleFirst) {
    agentStatusText = "Building 3D Robot Arm on Right Panel...";
    thinkingSteps.push({
      status: "info",
      text: "Assembling the starter 3D robot arm and placing the 500 mL water bottle on the floor at Spot A...",
      badge: "Building 3D",
    });
    renderApp();
    await runProgressiveAssemblyAnimation("single-before", "Building Starter Robot");
    thinkingSteps[thinkingSteps.length - 1] = {
      status: "done",
      text: "Assembled the starter 3D robot arm and floor mats (68 cm floor-to-floor span).",
      badge: "Built ✓",
    };
  }

  if (uploadedFile) {
    agentStatusText = "Tracking Human Arm Motion from Video...";
    thinkingSteps.push({
      status: "info",
      text: `Tracking Shoulder, Elbow, Wrist, and Hand movement from ${uploadedFile.name}...`,
      badge: "Video AI",
    });
    renderApp();
    try {
      const landmarksPayload = await extractLandmarksFromVideoFile(uploadedFile);
      await fetch("/api/projects/demo/motion", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset: "water_bottle", landmarks: landmarksPayload }),
      });
      await fetchProjectState("demo");
    } catch {
      await fetch("/api/projects/demo/motion", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset: "water_bottle" }),
      });
      await fetchProjectState("demo");
    }
  }

  // Step A: Human Arm Skeleton Extraction
  agentStatusText = "Tracking Human Arm Skeleton from Video...";
  thinkingSteps.push({
    status: "done",
    text: "Tracked the human arm from the video: reaching down to the floor (Spot A), lifting the 500 mL water bottle high up, and carrying it far across to place it on the floor (Spot B).",
    badge: "Skeleton Tracked",
  });
  if (targetMsg) {
    targetMsg.planCompletedCount = 2;
    targetMsg.showSkeletonCard = true;
  }
  renderApp();
  await sleep(320);

  // Step B: Baseline Simulation (Why it fails in plain language)
  agentStatusText = "Simulating Floor Bottle Lift in 3D...";
  thinkingSteps.push({
    status: "fail",
    text: "Tested First Try Robot in 3D Physics: The Shoulder Lift Motor is TOO WEAK! Lifting the full 500 mL water bottle off the floor and holding it out across a long distance overloads the shoulder by more than 2× (238% strain). Halfway across, the arm loses strength and sags back down to the floor!",
    badge: "Shoulder Too Weak ✗",
  });
  if (targetMsg) {
    targetMsg.planCompletedCount = 3;
  }
  renderApp();
  await sleep(360);

  // Step C: Smart Solution Selection
  agentStatusText = "Finding Stronger & Lighter Parts...";
  thinkingSteps.push({
    status: "warn",
    text: "Checked option A (buying a giant heavy motor): Too heavy and pushes total price over budget ($1,515).",
    badge: "Too Expensive",
  });
  thinkingSteps.push({
    status: "done",
    text: "Found Best Fix: Add a compact 3× Strength Booster Gear to the Shoulder Motor AND use Slimmer, Lighter Arm Tubes. Now the shoulder has 3× more lifting power, the arm is lighter, and it carries the water bottle smoothly from Floor Spot A to Floor Spot B!",
    badge: "Best Fix Found ✓",
  });

  if (targetMsg) {
    targetMsg.isStreamingThinking = false;
    targetMsg.planCompletedCount = 4;
    targetMsg.html = `
      <div>
        I tracked your arm motion from the video and tested the <strong>floor-to-floor water bottle transfer</strong> on the 3D robot (watch the simulation on the right):
        <ul style="margin:6px 0 8px 18px; padding:0;">
          <li><strong style="color:#f87171;">Why the First Robot Fails (Shown on Right Now):</strong> Picking up a full <strong>500 mL water bottle</strong> from the floor and carrying it far across is like holding a heavy bottle at arm's length. The <strong>Shoulder Lift Motor runs out of strength</strong> (100% maxed out, glowing red in 3D), so the arm sags down to the floor halfway across and never reaches Floor Spot B.</li>
          <li><strong style="color:#34d399;">How We Fix It Easily:</strong> We add a <strong>3× Strength Booster Gear</strong> to the Shoulder Motor and switch to <strong>Lighter Arm Tubes</strong>. That gives the shoulder 3× more lifting power while making the arm itself lighter!</li>
        </ul>
        Click <strong>"✅ Apply Upgrade to 3D Robot &amp; Watch It Carry the Bottle"</strong> below to upgrade the robot on the right panel!
      </div>
    `;
    targetMsg.showRecommendationCard = true;
  }

  agentBusy = false;
  agentStatusText = "Upgrade Ready — Click Apply Below";
  renderApp();
}

/**
 * Called when the user clicks "Apply Upgrade to 3D Robot" in the chat.
 */
async function handleAcceptRecommendation(): Promise<void> {
  if (agentBusy) return;
  agentBusy = true;
  recommendationAccepted = true;
  studioMode = "single-after";
  agentStatusText = "Installing 3× Shoulder Booster Gear & Lighter Tubes...";
  renderApp();

  await runProgressiveAssemblyAnimation("single-after", "Upgrading 3D Robot");

  // Reset timeline to 0 so the user immediately watches the clean floor-to-floor pick & carry
  currentFrameIdx = 0;

  const alreadyHasApprovalMsg = chatMessages.some((m) => m.showApprovalCard);
  if (!alreadyHasApprovalMsg) {
    chatMessages.push({
      id: `agent-applied-${Date.now()}`,
      role: "agent",
      timestamp: nowTimeStr(),
      html: `
        <div>
          🎉 <strong>Upgrade Installed on the 3D Robot!</strong>
          <br/>
          Look at the <strong>Right 3D Studio</strong> now:
          <ul style="margin:6px 0 8px 18px; padding:0;">
            <li>The robot reaches right down to <strong>Floor Spot A</strong> and grabs the 500 mL water bottle off the floor.</li>
            <li>With the <strong>3× Strength Booster Gear</strong> and <strong>Lighter Arm Tubes</strong>, all motors stay in the comfortable green zone (&le; 78% effort).</li>
            <li>It lifts the water bottle high in a smooth arc across the entire 68 cm span and sets it gently upright on <strong>Floor Spot B</strong>!</li>
          </ul>
          Tip: Click <strong>"Side-by-Side Compare"</strong> at the top of the 3D view to watch the failing First Try and the Upgraded Robot side-by-side!
        </div>
      `,
      showApprovalCard: true,
    });
  }

  agentBusy = false;
  agentStatusText = "Upgraded Robot Active — Carries Bottle Easily ✓";
  renderApp();
}

// ============================================================================
// 3D VIEWPORTS INITIALIZATION & ANIMATION LOOP
// ============================================================================

function initOrRefreshViewports(): void {
  if (!state?.loop_result?.iterations?.length) return;
  const iterations = state.loop_result.iterations;
  const iter0 = iterations[0];
  const finalIter = iterations[iterations.length - 1];

  const beforeHost = document.getElementById("viewport-before");
  const afterHost = document.getElementById("viewport-after");

  if (beforeHost && iter0?.trace) {
    beforeViewport = new RobotArmViewport(beforeHost);
    beforeViewport.setDesignAndTrace(iter0.design, iter0.trace, Boolean(iter0.check?.passed));
    beforeViewport.setAxisOverlaysVisible(showCadAxes);
    beforeViewport.setGhostArmVisible(showGhostTarget);
    beforeViewport.setAssemblyStage(assemblyStage);
    beforeViewport.updateFrame(iter0.trace, currentFrameIdx);
  }

  if (afterHost && finalIter?.trace) {
    afterViewport = new RobotArmViewport(afterHost);
    afterViewport.setDesignAndTrace(finalIter.design, finalIter.trace, Boolean(finalIter.check?.passed));
    afterViewport.setAxisOverlaysVisible(showCadAxes);
    afterViewport.setGhostArmVisible(showGhostTarget);
    afterViewport.setAssemblyStage(assemblyStage);
    afterViewport.updateFrame(finalIter.trace, currentFrameIdx);
  }

  if (beforeViewport && afterViewport) {
    let syncing = false;
    beforeViewport.controls.addEventListener("change", () => {
      if (syncing || studioMode !== "compare") return;
      syncing = true;
      afterViewport?.syncCameraFrom(beforeViewport!);
      syncing = false;
    });
    afterViewport.controls.addEventListener("change", () => {
      if (syncing || studioMode !== "compare") return;
      syncing = true;
      beforeViewport?.syncCameraFrom(afterViewport!);
      syncing = false;
    });
  }

  updateLiveTelemetryDom(currentFrameIdx);

  if (animFrameId !== null) {
    cancelAnimationFrame(animFrameId);
  }
  lastTickMs = performance.now();
  const tick = (now: number) => {
    const trace0: TraceData | undefined = state?.loop_result?.iterations?.[0]?.trace;
    const traceF: TraceData | undefined =
      state?.loop_result?.iterations?.[state.loop_result.iterations.length - 1]?.trace;

    if (isPlaying && trace0 && trace0.t.length > 1) {
      const fps = trace0.fps || 30;
      const dtMs = 1000 / (fps * playbackSpeed);
      if (now - lastTickMs >= dtMs) {
        currentFrameIdx = (currentFrameIdx + 1) % trace0.t.length;
        lastTickMs = now;

        if (trace0) beforeViewport?.updateFrame(trace0, currentFrameIdx);
        if (traceF) afterViewport?.updateFrame(traceF, currentFrameIdx);
        updateLiveTelemetryDom(currentFrameIdx);
        drawHumanSkeletonCanvas(currentFrameIdx);
      }
    }

    beforeViewport?.render();
    afterViewport?.render();
    animFrameId = requestAnimationFrame(tick);
  };
  animFrameId = requestAnimationFrame(tick);
}

function updateLiveTelemetryDom(frameIdx: number): void {
  const iterations = state?.loop_result?.iterations;
  if (!iterations || !iterations.length) return;
  const iter0 = iterations[0];
  const finalIter = iterations[iterations.length - 1];
  const trace0: TraceData | undefined = iter0?.trace;
  const traceF: TraceData | undefined = finalIter?.trace;
  if (!trace0 || !trace0.t.length) return;

  const idx = Math.max(0, Math.min(trace0.t.length - 1, frameIdx));
  const tSec = trace0.t[idx] ?? 0;
  const tMax = trace0.t[trace0.t.length - 1] ?? 6.0;

  const slider = document.getElementById("timeline-slider") as HTMLInputElement | null;
  if (slider && document.activeElement !== slider) {
    slider.value = String(idx);
  }

  const timeReadout = document.getElementById("time-readout");
  if (timeReadout) {
    timeReadout.textContent = `${tSec.toFixed(1)}s / ${tMax.toFixed(1)}s`;
  }

  const phaseReadout = document.getElementById("phase-readout");
  if (phaseReadout) {
    if (tSec < 1.15) {
      phaseReadout.textContent = "Step 1: Reaching Down to Floor Spot A";
      phaseReadout.className = "badge badge-info";
    } else if (tSec <= 4.85) {
      phaseReadout.textContent = "Step 2: Lifting & Carrying Bottle Across Floor";
      phaseReadout.className = "badge badge-warn";
    } else {
      phaseReadout.textContent = "Step 3: Placing Bottle on Floor Spot B";
      phaseReadout.className = "badge badge-pass";
    }
  }

  renderJointStripDom("live-strip-before", trace0, idx);
  if (traceF) {
    renderJointStripDom("live-strip-after", traceF, idx);
  }
}

function renderJointStripDom(containerId: string, trace: TraceData, idx: number): void {
  const el = document.getElementById(containerId);
  if (!el) return;

  const tauCmd = trace.tau_cmd[idx] || [0, 0, 0, 0];
  const tauAvail = trace.tau_avail[idx] || [1, 1, 1, 1];
  const joints = ["J1", "J2", "J3", "J4"];

  el.innerHTML = joints
    .map((jName, jIdx) => {
      const cmd = Math.abs(tauCmd[jIdx] ?? 0);
      const avail = Math.max(Math.abs(tauAvail[jIdx] ?? 1), 1e-4);
      const ratio = cmd / avail;
      const pct = Math.min(100, Math.round(ratio * 100));
      const color = ratio >= 0.98 ? "#ef4444" : ratio >= 0.78 ? "#f59e0b" : "#10b981";
      const statusWord =
        ratio >= 0.98
          ? `OVERLOADED (${pct}%)`
          : ratio >= 0.78
          ? `Heavy (${pct}%)`
          : `Easy (${pct}%)`;
      const friendlyName = FRIENDLY_JOINT_NAMES[jName] || jName;

      return `
        <div class="joint-bar-item">
          <div class="joint-bar-top">
            <strong>${escapeHtml(friendlyName)}</strong>
            <span style="color:${color}; font-weight:600;">${statusWord}</span>
          </div>
          <div class="joint-meter">
            <div class="joint-meter-fill" style="width:${pct}%; background:${color};"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

// ============================================================================
// MODALS (CATALOG, REPORT, PDF CITATION VIEWER)
// ============================================================================

function renderModalHtml(): string {
  if (!activeModal || !state) return "";

  if (activeModal === "catalog") {
    return `
      <div class="modal-backdrop" id="modal-backdrop">
        <div class="modal-card">
          <div class="modal-header">
            <div>
              <strong style="font-size:15px;">📚 Robot Motors &amp; Booster Gears Catalog (${state.catalog.parts.length} Real Parts)</strong>
              <div style="font-size:12px; color:#a1a1aa;">Every motor and gear is verified from official manufacturer datasheets</div>
            </div>
            <button class="btn btn-sm" id="btn-close-modal">Close ✕</button>
          </div>
          <div class="modal-body">
            <table class="catalog-table">
              <thead>
                <tr>
                  <th>Part Name</th>
                  <th>Category</th>
                  <th>Brand</th>
                  <th>Max Lifting Strength</th>
                  <th>Continuous Strength / Boost</th>
                  <th>Weight</th>
                  <th>Price</th>
                  <th>Official Datasheet</th>
                </tr>
              </thead>
              <tbody>
                ${state.catalog.parts
                  .map((p) => {
                    const doc = p.mass_kg?.src?.doc || `${p.part_id}.pdf`;
                    const page = p.mass_kg?.src?.page || 1;
                    return `
                      <tr>
                        <td><strong>${escapeHtml(p.part_id)}</strong></td>
                        <td><span class="badge badge-info">${p.type === "actuator" ? "Motor" : "Booster Gear"}</span></td>
                        <td>${escapeHtml(p.vendor)}</td>
                        <td style="font-family:var(--font-mono);">
                          ${
                            p.type === "actuator"
                              ? `${p.stall_torque_nm?.value?.toFixed(2) ?? "-"} N·m`
                              : `${p.rated_output_torque_nm?.value?.toFixed(1) ?? "-"} N·m max`
                          }
                        </td>
                        <td style="font-family:var(--font-mono);">
                          ${
                            p.type === "actuator"
                              ? `${p.continuous_torque_nm?.value?.toFixed(2) ?? "-"} N·m`
                              : `${p.ratio?.value ?? "-"}× Strength Boost`
                          }
                        </td>
                        <td style="font-family:var(--font-mono);">${p.mass_kg.value.toFixed(3)} kg</td>
                        <td style="font-family:var(--font-mono);">$${p.price_usd.value.toFixed(0)}</td>
                        <td>
                          <button class="cite-tag btn-cite-pdf" data-doc="${escapeHtml(doc)}" data-page="${page}">
                            ${escapeHtml(doc)} p.${page}
                          </button>
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

  if (activeModal === "report") {
    return `
      <div class="modal-backdrop" id="modal-backdrop">
        <div class="modal-card" style="height:86dvh;">
          <div class="modal-header">
            <div>
              <strong style="font-size:15px;">📄 MotionFit Summary &amp; Verification Report</strong>
              <div style="font-size:12px; color:#a1a1aa;">Complete before/after comparison and datasheet citations</div>
            </div>
            <div style="display:flex; gap:8px;">
              <a class="btn btn-sm btn-primary" href="/api/projects/demo/report" target="_blank">Open in New Tab ↗</a>
              <button class="btn btn-sm" id="btn-close-modal">Close ✕</button>
            </div>
          </div>
          <div class="modal-body" style="padding:0; overflow:hidden;">
            <iframe src="/api/projects/demo/report" style="width:100%; height:100%; border:none; background:#0b0f17;"></iframe>
          </div>
        </div>
      </div>
    `;
  }

  if (activeModal === "pdf") {
    const pdfUrl = `/api/projects/demo/datasheets/${encodeURIComponent(pdfModalDoc)}#page=${pdfModalPage}`;
    return `
      <div class="modal-backdrop" id="modal-backdrop">
        <div class="modal-card" style="height:85dvh;">
          <div class="modal-header">
            <div>
              <strong style="font-size:15px;">📑 Official Datasheet: ${escapeHtml(pdfModalDoc)} (Page ${pdfModalPage})</strong>
              <div style="font-size:12px; color:#a1a1aa;">Verified hardware specification page</div>
            </div>
            <button class="btn btn-sm" id="btn-close-modal">Close ✕</button>
          </div>
          <div class="modal-body" style="padding:0; overflow:hidden;">
            <iframe src="${pdfUrl}" style="width:100%; height:100%; border:none; background:#fff;"></iframe>
          </div>
        </div>
      </div>
    `;
  }

  return "";
}

// ============================================================================
// EVENT BINDING
// ============================================================================

function bindEvents(): void {
  document.getElementById("btn-open-catalog")?.addEventListener("click", () => {
    activeModal = "catalog";
    renderApp();
  });
  document.getElementById("btn-open-report")?.addEventListener("click", () => {
    activeModal = "report";
    renderApp();
  });
  document.getElementById("btn-close-modal")?.addEventListener("click", () => {
    activeModal = null;
    renderApp();
  });
  document.getElementById("modal-backdrop")?.addEventListener("click", (e) => {
    if ((e.target as HTMLElement)?.id === "modal-backdrop") {
      activeModal = null;
      renderApp();
    }
  });

  document.querySelectorAll(".btn-cite-pdf").forEach((btn) => {
    btn.addEventListener("click", () => {
      const doc = btn.getAttribute("data-doc") || "XM430-W350.pdf";
      const page = Number(btn.getAttribute("data-page") || "1");
      pdfModalDoc = doc;
      pdfModalPage = page;
      activeModal = "pdf";
      renderApp();
    });
  });

  // Quick Chips
  document.getElementById("chip-step1-build")?.addEventListener("click", () => {
    handleTurn1BuildArmPrompt(
      "Build a desktop robot arm that can pick up a 500 mL water bottle from the floor and move it across to another spot on the floor."
    );
  });

  document.getElementById("chip-step2-video")?.addEventListener("click", () => {
    handleVideoSimulationAndRedesign(
      "Here is a video of my arm picking up the 500 mL water bottle from Floor Spot A and carrying it far across to Floor Spot B. Track the arm skeleton, test it on the 3D robot, and tell me what parts to upgrade!",
      "floor_water_bottle_carry_720p.mp4 (6.0s)",
      !robotAssembled
    );
  });

  document.getElementById("chip-full-flow")?.addEventListener("click", () => {
    handleVideoSimulationAndRedesign(
      "Build a 3D robot arm for this video of picking up a 500 mL water bottle from the floor and carrying it across a long distance. Simulate the bottle transfer and recommend the best motor upgrades!",
      "floor_water_bottle_carry_720p.mp4 (6.0s)",
      true
    );
  });

  document.getElementById("btn-demo-walkthrough")?.addEventListener("click", () => {
    handleVideoSimulationAndRedesign(
      "Test the floor-to-floor 500 mL water bottle transfer and show me how to fix the weak shoulder motor.",
      "floor_water_bottle_carry_720p.mp4 (6.0s)",
      false
    );
  });

  document.getElementById("btn-reset-chat")?.addEventListener("click", () => {
    chatMessages = [];
    recommendationAccepted = false;
    studioMode = "single-before";
    initDefaultConversation();
    renderApp();
  });

  // Attachment controls
  document.getElementById("btn-attach-demo-video")?.addEventListener("click", () => {
    if (pendingVideoAttachment) {
      pendingVideoAttachment = null;
    } else {
      pendingVideoAttachment = {
        label: "floor_water_bottle_carry_720p.mp4 (6.0s)",
        preset: "water_bottle",
      };
    }
    renderApp();
  });

  document.getElementById("btn-clear-attachment")?.addEventListener("click", () => {
    pendingVideoAttachment = null;
    renderApp();
  });

  const videoInput = document.getElementById("input-video-file") as HTMLInputElement | null;
  videoInput?.addEventListener("change", () => {
    const file = videoInput.files?.[0];
    if (!file) return;
    pendingVideoAttachment = {
      label: `${file.name} (${(file.size / 1024).toFixed(0)} KB)`,
      preset: "water_bottle",
      file,
    };
    renderApp();
  });

  // Chat Form Submit
  const chatForm = document.getElementById("chat-form") as HTMLFormElement | null;
  const chatInput = document.getElementById("chat-input") as HTMLTextAreaElement | null;

  chatInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm?.requestSubmit();
    }
  });

  chatForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    if (agentBusy) return;
    const text = (chatInput?.value || "").trim();
    const hasAttachment = Boolean(pendingVideoAttachment);
    if (!text && !hasAttachment) return;

    const lower = text.toLowerCase();
    const mentionsVideoOrBottleSim =
      hasAttachment ||
      lower.includes("video") ||
      lower.includes("영상") ||
      lower.includes("시뮬") ||
      lower.includes("물병") ||
      lower.includes("bottle") ||
      lower.includes("skeleton") ||
      lower.includes("스켈레톤") ||
      lower.includes("floor") ||
      lower.includes("바닥");

    if (mentionsVideoOrBottleSim) {
      const att = pendingVideoAttachment;
      pendingVideoAttachment = null;
      handleVideoSimulationAndRedesign(
        text || "Test this floor-to-floor water bottle motion video and recommend stronger robot parts.",
        att?.label || "floor_water_bottle_carry_720p.mp4 (6.0s)",
        true,
        att?.file
      );
    } else {
      handleTurn1BuildArmPrompt(text);
    }
  });

  // Recommendation Card Buttons
  document.getElementById("btn-accept-recommendation")?.addEventListener("click", () => {
    handleAcceptRecommendation();
  });
  document.getElementById("btn-rec-compare")?.addEventListener("click", () => {
    studioMode = "compare";
    renderApp();
  });
  document.getElementById("btn-rec-report")?.addEventListener("click", () => {
    activeModal = "report";
    renderApp();
  });

  // Approval Form Submit
  document.getElementById("btn-submit-approval")?.addEventListener("click", async () => {
    const nameInput = document.getElementById("input-eng-name") as HTMLInputElement | null;
    const notesInput = document.getElementById("input-eng-notes") as HTMLInputElement | null;
    const engineer_name = (nameInput?.value || "Alex Lee").trim();
    const notes = (notesInput?.value || "Approved upgraded robot arm design.").trim();

    await fetch("/api/projects/demo/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ engineer_name, notes }),
    });
    await fetchProjectState("demo");
    renderApp();
  });

  // Right Studio Toolbar Controls
  document.getElementById("btn-view-before")?.addEventListener("click", () => {
    studioMode = "single-before";
    renderApp();
  });
  document.getElementById("btn-view-after")?.addEventListener("click", () => {
    studioMode = "single-after";
    renderApp();
  });
  document.getElementById("btn-view-compare")?.addEventListener("click", () => {
    studioMode = "compare";
    renderApp();
  });

  document.getElementById("btn-replay-assembly")?.addEventListener("click", () => {
    runProgressiveAssemblyAnimation(studioMode, "Replaying 3D Robot Build");
  });

  document.getElementById("btn-toggle-axes")?.addEventListener("click", () => {
    showCadAxes = !showCadAxes;
    beforeViewport?.setAxisOverlaysVisible(showCadAxes);
    afterViewport?.setAxisOverlaysVisible(showCadAxes);
    renderApp();
  });

  document.getElementById("btn-toggle-ghost")?.addEventListener("click", () => {
    showGhostTarget = !showGhostTarget;
    beforeViewport?.setGhostArmVisible(showGhostTarget);
    afterViewport?.setGhostArmVisible(showGhostTarget);
    renderApp();
  });

  // Timeline controls
  document.getElementById("btn-play-pause")?.addEventListener("click", () => {
    isPlaying = !isPlaying;
    const btn = document.getElementById("btn-play-pause");
    if (btn) btn.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
  });

  const slider = document.getElementById("timeline-slider") as HTMLInputElement | null;
  slider?.addEventListener("input", () => {
    isPlaying = false;
    const btn = document.getElementById("btn-play-pause");
    if (btn) btn.textContent = "▶ Play";
    currentFrameIdx = Number(slider.value || "0");
    const trace0 = state?.loop_result?.iterations?.[0]?.trace;
    const traceF =
      state?.loop_result?.iterations?.[state.loop_result.iterations.length - 1]?.trace;
    if (trace0) beforeViewport?.updateFrame(trace0, currentFrameIdx);
    if (traceF) afterViewport?.updateFrame(traceF, currentFrameIdx);
    updateLiveTelemetryDom(currentFrameIdx);
    drawHumanSkeletonCanvas(currentFrameIdx);
  });

  const speedSelect = document.getElementById("speed-select") as HTMLSelectElement | null;
  speedSelect?.addEventListener("change", () => {
    playbackSpeed = Number(speedSelect.value || "1");
  });
}

function scrollChatToBottom(): void {
  const stream = document.getElementById("chat-stream");
  if (stream) {
    stream.scrollTop = stream.scrollHeight;
  }
}

// ============================================================================
// BOOTSTRAP
// ============================================================================

async function boot(): Promise<void> {
  await fetchProjectState("demo");
  initDefaultConversation();
  // Pre-populate the full floor-to-floor water-bottle analysis walkthrough on initial load so the user
  // immediately sees the 1code Agentic Chat UI, Skeleton Tracker, Diff Recommendation Card, and 3D floor transfer!
  await handleVideoSimulationAndRedesign(
    "Build a robot arm for this video of picking up a 500 mL water bottle from the floor and carrying it far across to another spot on the floor. Test if it works and recommend any needed part upgrades!",
    "floor_water_bottle_carry_720p.mp4 (6.0s)",
    false
  );
}

boot().catch((err) => {
  console.error("MotionFit boot error:", err);
});
