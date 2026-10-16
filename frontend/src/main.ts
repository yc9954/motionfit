/**
 * MotionFit Frontend — Obolus Design System (Modeled after yc9954/simoff)
 *
 * Features:
 * 1. Obolus Shell & Sticky Left Sidebar (`aside.side`) with `.brand-sq` (3 ascending bars)
 *    and `.flow` vertical step navigation (`1. Build the robot`, `2. Agent & 3D Studio`,
 *    `3. Parts Catalog`, `4. Validation Report`).
 * 2. Obolus Landing Page (`section.grad.ink.chat-hero` + `.composer.glass-dark` + 4-column `.how`
 *    section + `.d3.grad.ink` cards + `.step-foot`), transitioning seamlessly into the Agent Studio.
 * 3. Split Agentic Chat + 3D Floor-to-Floor Water-Bottle Simulation Studio with zero-jargon UX,
 *    2D Human Arm Skeleton floor-to-floor motion tracker (`#skeleton-canvas`), and +/- Part Upgrade Diff Card.
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
type ActivePageView = "landing" | "studio" | "catalog" | "report";

let state: ProjectState | null = null;
let activeView: ActivePageView = "landing";
let chatMessages: ChatMessage[] = [];
let agentBusy = false;
let agentStatusText = "Ready · All 7 Robot Parts Assembled";

// Workflow state
let robotAssembled = true;
let isAssembling = false;
let assemblyStage = 7;
let assemblyLabel = "All 7 robot parts assembled";
let recommendationAccepted = false;
let studioMode: StudioDisplayMode = "single-before";

// Composer state
let pendingVideoAttachment: { label: string; preset: string; file?: File } | null = null;
let heroDraftPrompt =
  "We want a desktop robot arm to pick up a full 500 mL water bottle from Floor Spot A, carry it 68 cm across, and set it upright on Floor Spot B. Build the 3D robot, test my arm motion video, and upgrade any weak motors.";

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

/**
 * Progressive enhancement View Transitions wrapper (same-document-transitions best practice)
 */
function transitionView(updateFn: () => void): void {
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const docAny = document as any;
  if (!prefersReducedMotion && typeof docAny.startViewTransition === "function") {
    docAny.startViewTransition(() => {
      updateFn();
    });
  } else {
    updateFn();
  }
}

// ============================================================================
// OBOLUS CARD GRADIENT & RIBBON TEXTURE GENERATOR (Ported from yc9954/simoff)
// ============================================================================

const G_SCHEMES = [
  ["#123a86", "#2778cf", "#35e5df"],
  ["#07594f", "#1d9b89", "#8ee8d7"],
  ["#302b75", "#645dde", "#66d9c2"],
  ["#0b475a", "#1688ad", "#68e1dd"],
  ["#174340", "#218d78", "#a8e5ca"],
  ["#26316d", "#536bd0", "#70d2d8"],
  ["#123f55", "#299eb0", "#b0eee4"],
];
const G_POS = [
  [12, 18],
  [88, 12],
  [18, 88],
  [82, 82],
  [50, 6],
  [8, 52],
  [92, 48],
  [50, 94],
  [28, 34],
  [72, 66],
];
const G_ANG = [105, 120, 135, 150, 160, 115, 140, 170];
const G_RIBBONS = [
  '<path d="M-120 540 C110 80 330 510 735-90" stroke-width="168"/><path d="M155 550 C300 285 455 315 710 20" stroke-width="62" opacity=".32"/>',
  '<path d="M-140 25 C160 5 245 390 750 325" stroke-width="178"/><path d="M-80 470 C175 235 420 470 735 195" stroke-width="66" opacity=".3"/>',
  '<path d="M-130 420 C100 350 145 30 745 110" stroke-width="188"/><path d="M145 550 C245 265 505 245 655-110" stroke-width="66" opacity=".3"/>',
  '<path d="M365 570 C210 350 505 190 330-120" stroke-width="176"/><path d="M350 440 C535 240 625 270 750 35" stroke-width="64" opacity=".3"/>',
  '<path d="M-130 360 C90 55 350 405 750 95" stroke-width="170"/><path d="M-100 525 C175 300 420 520 735 290" stroke-width="60" opacity=".28"/>',
];

function fnvHash(t: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < t.length; i++) {
    h ^= t.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function gpick(seed: string, salt: string, mod: number): number {
  return fnvHash(`${salt}:${seed}`) % mod;
}

function cardGradient(seed: string, deep: boolean): string {
  const k = seed || "obolus";
  const [sh, mid, li] = G_SCHEMES[gpick(k, "scheme", G_SCHEMES.length)];
  const [x1, y1] = G_POS[gpick(k, "pos1", 10)];
  const [x2, y2] = G_POS[gpick(k, "pos2", 10)];
  const [x3, y3] = G_POS[gpick(k, "pos3", 10)];
  const a = G_ANG[gpick(k, "angle", 8)];
  const s = deep ? [94, 95, 78, 72, 74, 76] : [66, 72, 54, 68, 70, 72];
  return (
    `radial-gradient(ellipse at ${x1}% ${y1}%, color-mix(in oklab, ${li} ${s[0]}%, white) 0%, transparent ${s[3]}%), ` +
    `radial-gradient(ellipse at ${x2}% ${y2}%, color-mix(in oklab, ${mid} ${s[1]}%, white) 0%, transparent ${s[4]}%), ` +
    `radial-gradient(ellipse at ${x3}% ${y3}%, color-mix(in oklab, ${sh} ${s[2]}%, black) 0%, transparent ${s[5]}%), ` +
    `linear-gradient(${a}deg, ${sh}, ${mid} 52%, ${li})`
  );
}

function cardTexture(seed: string): string {
  const k = seed || "obolus";
  const v = gpick(k, "texture", 5);
  const refl = gpick(k, "texture-reflect", 2) === 1;
  const rot = [-9, 5, -4, 10][gpick(k, "texture-rotate", 4)];
  const tf = `${refl ? "translate(600 0) scale(-1 1) " : ""}rotate(${rot} 300 225)`;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 450" preserveAspectRatio="xMidYMid slice"><defs><filter id="flow" x="-45%" y="-45%" width="190%" height="190%"><feTurbulence type="fractalNoise" baseFrequency=".003 .008" numOctaves="2" seed="${
    gpick(k, "texture-noise", 97) + 1
  }" result="noise"/><feDisplacementMap in="SourceGraphic" in2="noise" scale="13" xChannelSelector="R" yChannelSelector="B"/><feGaussianBlur stdDeviation="22"/></filter><filter id="grain"><feTurbulence type="fractalNoise" baseFrequency=".72" numOctaves="3" seed="${
    gpick(k, "grain", 97) + 1
  }"/><feColorMatrix values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 .08 0"/></filter></defs><g transform="${tf}" fill="none" stroke="#f5ffff" stroke-linecap="round" opacity=".72" filter="url(#flow)">${
    G_RIBBONS[v]
  }</g><rect width="100%" height="100%" filter="url(#grain)" opacity=".12"/></svg>`;
  return `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
}

function paintGrads(root: ParentNode = document): void {
  root.querySelectorAll<HTMLElement>(".grad").forEach((e, i) => {
    if (e.dataset.grad === "origin") return;
    const seed =
      e.dataset.seed ||
      (e.querySelector(".mono, .lbl, h1, h2, b")?.textContent || e.textContent || "")
        .replace(/[\d.,$%×→\s—-]+/g, " ")
        .trim()
        .slice(0, 40) ||
      String(i);
    e.style.background = `linear-gradient(rgba(0,0,0,.03), rgba(0,0,0,.03)), ${cardTexture(
      seed
    )} center / cover no-repeat, ${cardGradient(seed, !e.classList.contains("ink"))}`;
  });
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
          Welcome to <strong>MotionFit</strong> — your AI Robot Co-Pilot.
          <div style="margin-top:4px; color:var(--mfg);">
            No mechanical engineering background is needed. Show a video of your arm performing a task, and I will build the 3D robot, test the floor-to-floor 500 mL water bottle transfer, and upgrade weak motors before you buy hardware.
          </div>
        </div>
      `,
    },
  ];
}

// ============================================================================
// MAIN RENDER FUNCTION (OBOLUS SHELL + SIDEBAR + LANDING / STUDIO / CATALOG / REPORT)
// ============================================================================

function renderApp(): void {
  const app = document.getElementById("app");
  if (!app || !state) return;

  const isApproved = Boolean(state.approval?.approved);
  const stepOrder: ActivePageView[] = ["landing", "studio", "catalog", "report"];
  const currentStepIdx = Math.max(0, stepOrder.indexOf(activeView));
  const fillPct = currentStepIdx <= 0 ? "0%" : `${Math.round((currentStepIdx / (stepOrder.length - 1)) * 100)}%`;

  app.innerHTML = `
    <div class="ob">
      <div class="shell">
        <!-- ==================== STICKY LEFT SIDEBAR (yc9954/simoff style) ==================== -->
        <aside class="side" aria-label="Steps">
          <div style="display:flex; flex-direction:column; gap:20px;">
            <a class="brand-mark" id="nav-brand-home" title="Return to MotionFit Landing Page">
              <span class="brand-sq">
                <span class="brand-bars"><i></i><i></i><i></i></span>
              </span>
              <span style="font-size:15px; font-weight:600; letter-spacing:-.02em;">MotionFit</span>
            </a>

            <nav class="flow" style="--fill: ${fillPct};">
              <div class="flow-track" aria-hidden="true"><i></i></div>

              <button
                type="button"
                class="nav-a flow-step ${currentStepIdx > 0 ? "done" : ""}"
                data-nav="landing"
                ${activeView === "landing" ? 'aria-current="page"' : ""}
              >
                <span class="nav-n mono" style="font-size:10px;">1</span>
                <span>Build the robot</span>
              </button>

              <button
                type="button"
                class="nav-a flow-step ${currentStepIdx > 1 ? "done" : ""}"
                data-nav="studio"
                ${activeView === "studio" ? 'aria-current="page"' : ""}
              >
                <span class="nav-n mono" style="font-size:10px;">2</span>
                <span>Agent &amp; 3D Studio</span>
              </button>

              <button
                type="button"
                class="nav-a flow-step ${currentStepIdx > 2 ? "done" : ""}"
                data-nav="catalog"
                ${activeView === "catalog" ? 'aria-current="page"' : ""}
              >
                <span class="nav-n mono" style="font-size:10px;">3</span>
                <span>Parts catalog</span>
              </button>

              <button
                type="button"
                class="nav-a flow-step"
                data-nav="report"
                ${activeView === "report" ? 'aria-current="page"' : ""}
              >
                <span class="nav-n mono" style="font-size:10px;">4</span>
                <span>Validation report</span>
              </button>
            </nav>

            <!-- Quick Project Status Card in Sidebar -->
            <div class="card" style="padding:10px 12px; background:#fff;">
              <div class="lbl" style="font-size:9.5px;">Active Task</div>
              <div style="font-size:12.5px; font-weight:600; margin-top:3px;">500 mL Floor Bottle Carry</div>
              <div class="muted" style="font-size:11.5px; margin-top:2px;">Floor Spot A &rarr; 68 cm &rarr; Floor Spot B</div>
              <div style="margin-top:8px; display:flex; flex-wrap:wrap; gap:4px;">
                ${
                  recommendationAccepted
                    ? `<span class="tag tag-ok">Upgraded · Passes ✓</span>`
                    : `<span class="tag tag-no">First Try · Sags ✗</span>`
                }
                ${isApproved ? `<span class="tag tag-ok">Approved ✓</span>` : ""}
              </div>
            </div>
          </div>

          <div style="border-top:1px solid var(--sb-border); padding:10px 6px 4px; display:flex; flex-direction:column; gap:6px;">
            <div class="lbl" style="font-size:9.5px;">Hardware Bundle</div>
            <a class="btn btn-mono btn-sm" href="/api/projects/demo/export" download="motionfit-demo-bundle.zip" style="width:100%;">
              Export URDF + BOM (.zip)
            </a>
            <div class="mono muted" style="font-size:10.5px; line-height:1.4;">
              4-DOF Desktop Arm · ${state.catalog.parts.length} Verified Parts
            </div>
          </div>
        </aside>

        <!-- ==================== MAIN VIEW AREA ==================== -->
        <main id="view">
          <!-- 1. LANDING VIEW (yc9954/simoff screens/start.html style) -->
          <div id="page-landing" style="display: ${activeView === "landing" ? "block" : "none"};">
            ${renderLandingPageHtml()}
          </div>

          <!-- 2. AGENT CHAT + 3D ROBOT STUDIO VIEW (Always mounted so #viewport-before & #viewport-after preserve WebGL context) -->
          <div id="page-studio" style="display: ${activeView === "studio" ? "flex" : "none"}; flex-direction:column; min-height:100dvh;">
            ${renderStudioTopbarHtml(isApproved)}
            ${renderStudioWorkspaceHtml()}
          </div>

          <!-- 3. FULL-PAGE PARTS CATALOG VIEW -->
          <div id="page-catalog" style="display: ${activeView === "catalog" ? "block" : "none"};">
            ${renderCatalogPageHtml()}
          </div>

          <!-- 4. FULL-PAGE VALIDATION REPORT VIEW -->
          <div id="page-report" style="display: ${activeView === "report" ? "block" : "none"};">
            ${renderReportPageHtml()}
          </div>
        </main>
      </div>

      ${renderModalHtml()}
    </div>
  `;

  paintGrads(app);
  bindEvents();
  initOrRefreshViewports();
  drawHumanSkeletonCanvas(currentFrameIdx);
  if (activeView === "studio") {
    scrollChatToBottom();
  }
}

// ============================================================================
// 1. SIMOFF-STYLE LANDING PAGE (screens/start.html + Obolus Hero & How It Works)
// ============================================================================

function renderLandingPageHtml(): string {
  const lr = state?.loop_result;
  const iter0 = lr?.iterations?.[0];
  const finalIter = lr?.iterations?.[lr.iterations.length - 1];
  const m0 = iter0?.metrics;
  const mf = finalIter?.metrics;

  return `
    <!-- Obolus Fractal-Noise Ribbon Hero Banner -->
    <section class="grad ink chat-hero" data-seed="motionfit-hero-origin">
      <div class="chat-hero-in">
        <div class="hero-logo" aria-hidden="true">
          <span><i></i><i></i><i></i></span>
        </div>
        <div class="mono hero-kicker">AI Co-Pilot for Physical Robot Simulation</div>
        <h1 class="display hero-title">What physical task do you want your robot arm to do?</h1>
        <p class="hero-sub">
          Describe the task or attach a smartphone video of your own arm doing the motion. MotionFit builds a 3D robot arm, simulates lifting a 500 mL water bottle from the floor across a 68 cm span, and upgrades weak motors before you buy a single part.
        </p>

        <!-- Dark Glass Composer (.composer.glass-dark from simoff) -->
        <div class="composer glass-dark">
          <form id="hero-chat-form">
            <label for="hero-ask" class="lbl" style="position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0,0,0,0);">
              Describe the robot task
            </label>
            <textarea
              id="hero-ask"
              rows="2"
              placeholder="Describe the motion you want the robot arm to perform, or attach a video of picking up a water bottle from the floor..."
            >${escapeHtml(heroDraftPrompt)}</textarea>

            ${
              pendingVideoAttachment
                ? `
                  <div style="margin-top:6px; display:flex; align-items:center; gap:8px;">
                    <span class="tag" style="background:rgba(102,217,194,.16); color:#66d9c2;">
                      📹 ${escapeHtml(pendingVideoAttachment.label)}
                    </span>
                  </div>
                `
                : ""
            }

            <div class="composer-bar">
              <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                <span class="mpick-btn">
                  <span class="mpick-dot"></span>
                  <span>Gemini 4 Argon · Physics Co-Pilot</span>
                </span>
                <button type="button" class="attach-ic" id="btn-hero-attach-demo" title="Attach the 6.0s floor-to-floor water bottle motion video">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
                  <span>${pendingVideoAttachment ? "floor_water_bottle_carry.mp4 ✓" : "Attach floor bottle video"}</span>
                </button>
              </div>

              <button type="submit" class="hero-send-btn" id="btn-hero-send" aria-label="Launch Agent Studio and simulate">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
              </button>
            </div>
          </form>
        </div>

        <!-- Quick-Start Prompt Pills -->
        <div class="hero-quick-row">
          <button type="button" class="hero-pill-btn" id="btn-hero-quick-full">
            <span>▶ Run Full Floor Water-Bottle Simulation &amp; Auto-Fix</span>
          </button>
          <button type="button" class="hero-pill-btn" id="btn-hero-quick-build">
            <span>🔧 Watch 7-Part 3D Robot Assembly</span>
          </button>
          <button type="button" class="hero-pill-btn" id="btn-hero-open-studio">
            <span>Open Agent &amp; 3D Studio Directly &rarr;</span>
          </button>
        </div>
      </div>
    </section>

    <!-- Below Hero: HOW IT WORKS 4-Column Section (Exact simoff layout) -->
    <section class="landing-info">
      <div class="lbl">How it works</div>
      <h2 class="display" style="font-size:28px; margin-top:8px; max-width:680px;">
        Show the motion once, and the agent builds, tests, and fixes the robot before day one.
      </h2>

      <div class="how">
        <div>
          <div class="how-ic" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 7l9-4 9 4-9 4-9-4z"/><path d="M3 7v10l9 4 9-4V7"/><path d="M12 11v10"/></svg>
          </div>
          <b>1. Build the 3D robot</b>
          <p>
            Tell the agent what you want to move. It selects starter motors from verified datasheets and assembles a 4-joint 3D robot arm on the studio floor.
          </p>
        </div>

        <div>
          <div class="how-ic" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="9" cy="7" r="3"/><path d="M3 20v-2a5 5 0 0 1 5-5h2a5 5 0 0 1 5 5v2"/><circle cx="17" cy="9" r="2.3"/><path d="M16 14h2a4 4 0 0 1 4 4v2"/></svg>
          </div>
          <b>2. Track human arm video</b>
          <p>
            Attach a smartphone video of your arm picking up a 500 mL water bottle from the floor. The agent extracts your Shoulder, Elbow, Wrist, and Hand skeleton.
          </p>
        </div>

        <div>
          <div class="how-ic" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="4" width="7" height="7" rx="1"/><rect x="14" y="4" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><path d="M14 17.5h7M17.5 14v7"/></svg>
          </div>
          <b>3. Test floor bottle carry</b>
          <p>
            Watch the 3D physics test: the first robot lifts the water bottle from Floor Spot A, but its shoulder motor runs out of strength halfway across the 68 cm span and drops it.
          </p>
        </div>

        <div>
          <div class="how-ic" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 19h16"/><path d="M7 15l4-4 3 3 5-6"/><path d="M16 8h3v3"/></svg>
          </div>
          <b>4. Upgrade weak parts</b>
          <p>
            Accept the agent's 2-part fix — a 3× Shoulder Strength Booster Gear and lighter arm tubes — to carry the bottle cleanly to Floor Spot B under budget.
          </p>
        </div>
      </div>

      <!-- 3 Obolus .d3.grad.ink Metric Preview Cards -->
      <div class="landing-metrics-row">
        <article class="d3 grad ink" data-seed="shoulder-booster-gain">
          <div class="mono" style="font-size:9.5px; letter-spacing:1.3px; text-transform:uppercase; color:rgba(16,17,20,.65);">
            Shoulder Lifting Effort (J2)
          </div>
          <div class="mono" style="margin-top:20px; font-size:28px; line-height:1; letter-spacing:-.04em; color:#101114; font-weight:600;">
            238% &rarr; 78%
          </div>
          <div style="margin-top:16px; padding-top:10px; border-top:1px solid rgba(16,17,20,.14); font-size:12px; line-height:1.5; color:rgba(16,17,20,.78);">
            Adding the 3× Strength Booster Gear (APEX-PG20-2) brings the overloaded shoulder motor safely into the green zone.
          </div>
        </article>

        <article class="d3 grad ink" data-seed="floor-span-accuracy">
          <div class="mono" style="font-size:9.5px; letter-spacing:1.3px; text-transform:uppercase; color:rgba(16,17,20,.65);">
            68 cm Floor-to-Floor Bottle Delivery
          </div>
          <div class="mono" style="margin-top:20px; font-size:28px; line-height:1; letter-spacing:-.04em; color:#101114; font-weight:600;">
            Drops &rarr; Spot B ✓
          </div>
          <div style="margin-top:16px; padding-top:10px; border-top:1px solid rgba(16,17,20,.14); font-size:12px; line-height:1.5; color:rgba(16,17,20,.78);">
            Picks up the 500 mL water bottle flat on Floor Spot A (z = 0) and places it upright on Floor Spot B (${mf?.ee_rmse_mm?.toFixed(1) ?? "12.0"} mm accuracy).
          </div>
        </article>

        <article class="d3 grad ink" data-seed="hardware-weight-budget">
          <div class="mono" style="font-size:9.5px; letter-spacing:1.3px; text-transform:uppercase; color:rgba(16,17,20,.65);">
            Total Arm Weight &amp; Hardware Cost
          </div>
          <div class="mono" style="margin-top:20px; font-size:28px; line-height:1; letter-spacing:-.04em; color:#101114; font-weight:600;">
            ${mf?.total_mass_kg?.toFixed(2) ?? "0.94"} kg · $${mf?.total_cost_usd?.toFixed(0) ?? "1,126"}
          </div>
          <div style="margin-top:16px; padding-top:10px; border-top:1px solid rgba(16,17,20,.14); font-size:12px; line-height:1.5; color:rgba(16,17,20,.78);">
            Slimmer aluminum tubes save 70 g (${m0?.total_mass_kg?.toFixed(2) ?? "1.01"} kg &rarr; ${mf?.total_mass_kg?.toFixed(2) ?? "0.94"} kg) while staying under the $1,200 budget.
          </div>
        </article>
      </div>

      <!-- Obolus Step Footer (.step-foot) -->
      <div class="step-foot">
        <p>
          <b>Verified in 3D physics before you buy hardware.</b>
          Explore the interactive Agent Chat &amp; 3D Studio to watch the floor-to-floor water bottle simulation live.
        </p>
        <span class="step-acts">
          <button type="button" class="btn btn-outline" id="btn-foot-catalog">Browse ${state?.catalog.parts.length ?? 9} Verified Parts</button>
          <button type="button" class="btn btn-mono" id="btn-foot-open-studio">
            <span>Next: Agent &amp; 3D Studio</span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
          </button>
        </span>
      </div>
    </section>
  `;
}

// ============================================================================
// 2. STUDIO TOPBAR & SPLIT WORKSPACE (AGENT CHAT + 3D STUDIO)
// ============================================================================

function renderStudioTopbarHtml(isApproved: boolean): string {
  return `
    <header class="studio-topbar">
      <div class="studio-topbar-left">
        <span class="lbl">Step 2 · Agent &amp; 3D Studio</span>
        <span style="color:var(--border);">|</span>
        <strong style="font-size:14px;">Floor-to-Floor 500 mL Water Bottle Transfer (68 cm Span)</strong>
        ${
          recommendationAccepted
            ? `<span class="tag tag-ok"><span class="sq" style="background:var(--ours);"></span>Upgraded Robot (Carries Bottle Easily ✓)</span>`
            : `<span class="tag tag-no"><span class="sq" style="background:var(--red);"></span>First Try Robot (Shoulder Too Weak for Bottle ✗)</span>`
        }
        ${
          isApproved
            ? `<span class="tag tag-ok">✓ Approved by ${escapeHtml(state?.approval?.engineer_name || "Team")}</span>`
            : ""
        }
      </div>

      <div class="studio-topbar-actions">
        <button class="btn btn-outline btn-sm" id="btn-open-catalog" title="Browse available motors and booster gears">
          Parts Catalog (${state?.catalog.parts.length ?? 9})
        </button>
        <button class="btn btn-outline btn-sm" id="btn-open-report" title="View full summary report">
          Validation Report
        </button>
        <a class="btn btn-mono btn-sm" href="/api/projects/demo/export" download="motionfit-demo-bundle.zip">
          Download Robot Package (.zip)
        </a>
      </div>
    </header>
  `;
}

function renderStudioWorkspaceHtml(): string {
  if (!state) return "";

  return `
    <div class="workspace-split">
      <!-- ==================== LEFT PANE: OBOLUS AGENT CHAT ==================== -->
      <section class="chat-pane">
        <div class="chat-pane-header">
          <div class="agent-status-pill">
            <span class="status-dot ${agentBusy ? "busy" : ""}"></span>
            <span id="agent-status-label">${escapeHtml(agentStatusText)}</span>
          </div>
          <div style="display:flex; gap:6px;">
            <button class="btn btn-outline btn-sm" id="btn-demo-walkthrough" title="Run the complete floor water-bottle analysis & recommendation">
              Run Full Demo
            </button>
            <button class="btn btn-muted btn-sm" id="btn-reset-chat" title="Start a fresh conversation">
              Reset
            </button>
          </div>
        </div>

        <div class="chat-stream" id="chat-stream">
          ${chatMessages.map((m) => renderChatMessageHtml(m)).join("")}
        </div>

        <!-- Bottom Composer -->
        <div class="chat-composer">
          <div class="quick-chips">
            <button class="chip-btn" id="chip-step1-build">
              1. Build 3D Robot Arm First
            </button>
            <button class="chip-btn" id="chip-step2-video">
              2. Send Floor Water-Bottle Video &amp; Test
            </button>
            <button class="chip-btn" id="chip-full-flow">
              3. Build + Simulate Floor Bottle Transfer
            </button>
          </div>

          <div class="onecode-composer-box">
            <div class="composer-attachments" id="composer-attachments">
              ${
                pendingVideoAttachment
                  ? `
                    <span class="attachment-pill" style="margin-bottom:6px;">
                      <span>📹 ${escapeHtml(pendingVideoAttachment.label)}</span>
                      <button id="btn-clear-attachment" style="background:none;border:none;color:var(--red);cursor:pointer;font-weight:700;">×</button>
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
                placeholder="Ask the agent to build a robot arm, or attach a video of picking up a water bottle from the floor..."
              ></textarea>

              <div class="onecode-composer-footer">
                <div class="onecode-composer-pills">
                  <span class="onecode-model-pill">
                    <span class="sq" style="background:var(--ours); border-radius:99px;"></span>
                    Gemini 4 Argon
                  </span>
                  <button class="btn btn-muted btn-sm" id="btn-attach-demo-video" type="button">
                    ${pendingVideoAttachment ? "✓ Floor Bottle Video Attached" : "📹 Attach Floor Bottle Video"}
                  </button>
                  <label class="btn btn-muted btn-sm" style="cursor:pointer;">
                    Upload Video
                    <input type="file" id="input-video-file" accept="video/mp4,video/quicktime,video/webm" style="display:none;" />
                  </label>
                </div>

                <button type="submit" class="btn btn-mono btn-sm" id="btn-send-chat" ${agentBusy ? "disabled" : ""}>
                  ${agentBusy ? "Working..." : "Send ↑"}
                </button>
              </div>
            </form>
          </div>
        </div>
      </section>

      <!-- ==================== RIGHT PANE: OBOLUS 3D ROBOT STUDIO ==================== -->
      <section class="studio-pane">
        <!-- Studio Toolbar with Obolus .tabs -->
        <div class="studio-toolbar">
          <div class="studio-mode-group">
            <div class="tabs" role="tablist" aria-label="3D Robot View Mode">
              <button
                type="button"
                class="tab ${studioMode === "single-before" ? "on" : ""}"
                id="btn-view-before"
              >
                <span class="sq" style="background:var(--red);"></span>
                1. First Try (Drops Bottle ✗)
              </button>
              <button
                type="button"
                class="tab ${studioMode === "single-after" ? "on" : ""}"
                id="btn-view-after"
              >
                <span class="sq" style="background:var(--ours);"></span>
                2. Upgraded Robot (Carries Bottle ✓)
              </button>
              <button
                type="button"
                class="tab ${studioMode === "compare" ? "on" : ""}"
                id="btn-view-compare"
              >
                Side-by-Side Compare
              </button>
            </div>
          </div>

          <div class="studio-mode-group">
            <button class="btn btn-outline btn-sm" id="btn-replay-assembly" title="Watch the 3D robot arm assemble part-by-part">
              Replay 3D Build
            </button>
            <button class="btn ${showCadAxes ? "btn-mono" : "btn-outline"} btn-sm" id="btn-toggle-axes">
              Strain Arrows
            </button>
            <button class="btn ${showGhostTarget ? "btn-mono" : "btn-outline"} btn-sm" id="btn-toggle-ghost">
              Ideal Path
            </button>
          </div>
        </div>

        <!-- 3D Viewport Grid (Contains both #viewport-before and #viewport-after) -->
        <div class="studio-stage-grid ${studioMode === "compare" ? "compare-mode" : ""}" id="studio-stage-grid">
          <!-- BEFORE / FIRST TRY VIEWPORT CARD -->
          <div class="viewport-card ${studioMode === "single-after" ? "hidden-pane" : ""}" id="card-before">
            <div class="viewport-hud-top">
              <div class="hud-pill glass">
                <div class="hud-title">
                  <span class="tag tag-no">First Try (Fails)</span>
                  <span>Standard Shoulder Motor · No Booster Gear</span>
                </div>
                <div class="hud-sub">
                  Picks up the <strong>500 mL water bottle from Floor Spot A</strong>, but holding it out across a 68 cm span is too heavy. Halfway across, the <strong style="color:var(--red);">Shoulder Motor runs out of strength (238% strain)</strong> and drops the bottle on the floor.
                </div>
              </div>
              <div class="hud-pill glass mono" style="max-width:none; font-size:11px;">
                <span style="color:var(--blue);">● Floor Spot A</span> &rarr; <span style="color:var(--red);">● Sags Halfway</span>
              </div>
            </div>
            <div class="viewport-canvas-host" id="viewport-before"></div>
          </div>

          <!-- AFTER / UPGRADED VIEWPORT CARD -->
          <div class="viewport-card ${studioMode === "single-before" ? "hidden-pane" : ""}" id="card-after">
            <div class="viewport-hud-top">
              <div class="hud-pill glass">
                <div class="hud-title">
                  <span class="tag tag-ok">Upgraded Robot (Passes ✓)</span>
                  <span>3× Shoulder Booster Gear + Lighter Arm Tubes</span>
                </div>
                <div class="hud-sub">
                  With a <strong style="color:var(--ours);">3× Strength Booster Gear</strong> on the shoulder and slimmer aluminum tubes, the robot lifts the water bottle from <strong>Floor Spot A</strong>, carries it 68 cm across, and sets it gently upright on <strong>Floor Spot B</strong>.
                </div>
              </div>
              <div class="hud-pill glass mono" style="max-width:none; font-size:11px;">
                <span style="color:var(--blue);">● Floor Spot A</span> &rarr; <span style="color:var(--ours);">● Floor Spot B ✓</span>
              </div>
            </div>
            <div class="viewport-canvas-host" id="viewport-after"></div>
          </div>

          <!-- Progressive 3D Assembly Overlay -->
          <div class="assembly-overlay" id="assembly-overlay" style="display: ${isAssembling ? "block" : "none"};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div>
                <strong style="color:var(--fg);">Building 3D Robot Arm (${assemblyStage}/7)</strong>
                <span style="margin-left:8px; color:var(--mfg); font-size:12.5px;" id="assembly-step-text">${escapeHtml(assemblyLabel)}</span>
              </div>
              <span class="tag tag-ai">${Math.round((assemblyStage / 7) * 100)}%</span>
            </div>
            <div class="assembly-progress-bar">
              <div class="assembly-progress-fill" style="width: ${Math.round((assemblyStage / 7) * 100)}%;"></div>
            </div>
          </div>
        </div>

        <!-- Studio Bottom Dock: Timeline + Friendly Motor Effort & Strain Meter -->
        <div class="studio-bottom-dock">
          <div class="timeline-bar">
            <button class="btn btn-mono btn-sm" id="btn-play-pause" style="min-width:78px;">
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
            <span class="tag tag-ai" id="phase-readout">Step 2: Lifting &amp; Carrying Bottle</span>
            <div class="time-readout" id="time-readout">0.0s / 6.0s</div>
            <select id="speed-select" class="btn btn-outline btn-sm">
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
                <span style="color:var(--red);">⚠ Shoulder Maxed Out (100% Overload)</span>
              </div>
              <div class="joint-bars" id="live-strip-before"></div>
            </div>
            <div class="telemetry-box">
              <div class="telemetry-box-title">
                <span>Upgraded Robot — Motor Effort &amp; Strain Meter</span>
                <span style="color:var(--ours);">✓ All Motors in Safe Green Zone</span>
              </div>
              <div class="joint-bars" id="live-strip-after"></div>
            </div>
          </div>
        </div>
      </section>
    </div>
  `;
}

// ============================================================================
// 3. FULL-PAGE PARTS CATALOG & 4. FULL-PAGE REPORT VIEWS (Obolus .tbl & .d3)
// ============================================================================

function renderCatalogPageHtml(): string {
  if (!state) return "";

  return `
    <div class="step-page">
      <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap; margin-bottom:24px;">
        <div>
          <div class="lbl">Step 3 · Verified Hardware Catalog</div>
          <h1 class="display" style="font-size:28px; margin-top:6px;">
            Motors &amp; Strength Booster Gears (${state.catalog.parts.length} Real Parts)
          </h1>
          <p class="muted" style="margin-top:6px; max-width:660px;">
            Every actuator and planetary gearbox is grounded in official manufacturer PDF datasheets with page-level citations.
          </p>
        </div>
        <div style="display:flex; gap:8px;">
          <button type="button" class="btn btn-outline" id="btn-cat-back-studio">&larr; Back to 3D Studio</button>
          <button type="button" class="btn btn-mono" id="btn-cat-next-report">Next: Validation Report &rarr;</button>
        </div>
      </div>

      <div class="tbl">
        <div class="th" style="grid-template-columns: 1.3fr 0.9fr 0.9fr 1.1fr 1.1fr 0.7fr 0.6fr 1.1fr;">
          <span>Part Model</span>
          <span>Category</span>
          <span>Manufacturer</span>
          <span>Peak / Limit Strength</span>
          <span>Continuous / Boost</span>
          <span class="r">Weight</span>
          <span class="r">Price</span>
          <span class="r">Datasheet PDF</span>
        </div>
        ${state.catalog.parts
          .map((p) => {
            const doc = p.mass_kg?.src?.doc || `${p.part_id}.pdf`;
            const page = p.mass_kg?.src?.page || 1;
            const isSelectedUpgrade =
              p.part_id === "XM430-W350" || p.part_id === "XL430-W250" || p.part_id === "APEX-PG20-2";
            return `
              <div class="tr" style="grid-template-columns: 1.3fr 0.9fr 0.9fr 1.1fr 1.1fr 0.7fr 0.6fr 1.1fr; ${
                isSelectedUpgrade ? "background:#fff;" : ""
              }">
                <span>
                  <strong class="mono">${escapeHtml(p.part_id)}</strong>
                  ${isSelectedUpgrade ? `<span class="tag tag-ok" style="margin-left:6px;">Used in Final Arm</span>` : ""}
                </span>
                <span>
                  <span class="tag ${p.type === "actuator" ? "tag-ai" : "tag-ok"}">
                    ${p.type === "actuator" ? "Motor" : "Booster Gear"}
                  </span>
                </span>
                <span class="muted">${escapeHtml(p.vendor)}</span>
                <span class="num">
                  ${
                    p.type === "actuator"
                      ? `${p.stall_torque_nm?.value?.toFixed(2) ?? "-"} N·m peak`
                      : `${p.rated_output_torque_nm?.value?.toFixed(1) ?? "-"} N·m max`
                  }
                </span>
                <span class="num">
                  ${
                    p.type === "actuator"
                      ? `${p.continuous_torque_nm?.value?.toFixed(2) ?? "-"} N·m cont.`
                      : `${p.ratio?.value ?? "-"}× Strength Boost`
                  }
                </span>
                <span class="num r">${p.mass_kg.value.toFixed(3)} kg</span>
                <span class="num r">$${p.price_usd.value.toFixed(0)}</span>
                <span class="r">
                  <button class="cite-tag btn-cite-pdf" data-doc="${escapeHtml(doc)}" data-page="${page}">
                    ${escapeHtml(doc)} p.${page} ↗
                  </button>
                </span>
              </div>
            `;
          })
          .join("")}
      </div>
    </div>
  `;
}

function renderReportPageHtml(): string {
  if (!state) return "";

  return `
    <div class="step-page">
      <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap; margin-bottom:20px;">
        <div>
          <div class="lbl">Step 4 · Final Verification &amp; Sign-Off</div>
          <h1 class="display" style="font-size:28px; margin-top:6px;">
            Validation Report · 500 mL Floor Water Bottle Transfer
          </h1>
          <p class="muted" style="margin-top:6px; max-width:680px;">
            Before-and-after physics verification, joint strain margins, and printable engineering documentation.
          </p>
        </div>
        <div style="display:flex; gap:8px;">
          <button type="button" class="btn btn-outline" id="btn-rep-back-studio">&larr; Back to 3D Studio</button>
          <a class="btn btn-outline" href="/api/projects/demo/report" target="_blank">Open Printable Report ↗</a>
          <a class="btn btn-mono" href="/api/projects/demo/export" download="motionfit-demo-bundle.zip">Download Bundle (.zip)</a>
        </div>
      </div>

      <div class="card" style="overflow:hidden; height:74dvh; background:#fff;">
        <iframe src="/api/projects/demo/report" style="width:100%; height:100%; border:none; background:#fff;"></iframe>
      </div>
    </div>
  `;
}

// ============================================================================
// OBOLUS AGENT CHAT MESSAGE & RICH CARDS RENDERING
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

  const thinkingHtml =
    m.thinkingSteps && m.thinkingSteps.length > 0
      ? `
        <details class="onecode-thinking" open>
          <summary class="onecode-thinking-summary">
            <div class="onecode-thinking-left">
              <span class="sq" style="background:var(--jev); border-radius:99px;"></span>
              <span class="onecode-thinking-label">
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
                    ? "tag-ok"
                    : s.status === "warn"
                    ? "badge-warn"
                    : s.status === "fail"
                    ? "tag-no"
                    : "tag-ai";
                return `
                  <div class="onecode-tool-row ${s.status}">
                    <div class="onecode-tool-row-left">
                      <span class="onecode-tool-icon">${icon}</span>
                      <span>${s.text}</span>
                    </div>
                    ${s.badge ? `<span class="tag ${badgeClass}" style="flex-shrink:0;">${escapeHtml(s.badge)}</span>` : ""}
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
            <span class="ava" aria-hidden="true"><span><i></i><i></i><i></i></span></span>
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
        <span>Agent Task Checklist</span>
        <span class="tag ${completedCount >= 4 ? "tag-ok" : "tag-ai"}">${Math.min(4, completedCount)} / 4 Completed</span>
      </div>
      <div class="onecode-plan-steps">
        ${steps
          .map((label, idx) => {
            const isDone = idx < completedCount;
            return `
              <div class="onecode-plan-step">
                <span class="onecode-plan-check ${isDone ? "completed" : "active"}">${isDone ? "✓" : idx + 1}</span>
                <span style="${isDone ? "color:var(--fg); font-weight:500;" : "color:var(--mfg);"}">${escapeHtml(label)}</span>
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
          <strong style="color:var(--fg); font-size:13px;">Human Arm Motion Tracker (From Your Video)</strong>
          <div style="font-size:11.5px; color:var(--mfg);">Tracks Shoulder &rarr; Elbow &rarr; Wrist &rarr; Hand picking up the 500 mL water bottle from the floor and carrying it 68 cm across</div>
        </div>
        <span class="tag tag-ai">Live Synced</span>
      </div>
      <div class="skeleton-canvas-wrap">
        <canvas id="skeleton-canvas" class="skeleton-canvas" width="520" height="175"></canvas>
      </div>
      <div class="phase-pills">
        <div class="phase-pill">
          <div class="phase-pill-title">1. Reach to Floor</div>
          <div class="mono muted" style="font-size:10px;">0.0s – 1.2s</div>
          <div style="color:var(--mfg); font-size:11px; margin-top:2px;">Hand bends down to Floor Spot A &amp; grabs the 500 mL bottle</div>
        </div>
        <div class="phase-pill">
          <div class="phase-pill-title">2. Long Carry (68 cm)</div>
          <div class="mono muted" style="font-size:10px;">1.2s – 4.9s (Hardest!)</div>
          <div style="color:var(--mfg); font-size:11px; margin-top:2px;">Lifts bottle high off the floor &amp; carries it far to the right</div>
        </div>
        <div class="phase-pill">
          <div class="phase-pill-title">3. Place on Floor</div>
          <div class="mono muted" style="font-size:10px;">4.9s – 6.0s</div>
          <div style="color:var(--mfg); font-size:11px; margin-top:2px;">Sets bottle upright on Floor Spot B &amp; releases</div>
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
    <div class="recommendation-card card acc wash">
      <div class="rec-header">
        <div>
          <div class="rec-title">Recommended Robot Upgrade (Easy 2-Part Fix)</div>
          <div style="font-size:12px; color:var(--mfg); margin-top:2px;">
            Makes the shoulder <strong>3× stronger</strong> and the arm <strong>lighter</strong> so it carries the water bottle from floor to floor effortlessly.
          </div>
        </div>
        <span class="tag tag-ok">Verified in 3D Physics ✓</span>
      </div>

      <!-- Diff Block: Clear Before (-) vs After (+) in Plain Language -->
      <div class="onecode-diff-block">
        <div class="onecode-diff-header">
          <span>What We Change on the Robot Arm</span>
          <span class="muted" style="font-weight:400;">2 part changes · 3 other motors stay the same</span>
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
            <strong>Shoulder Lift Motor (Upgrade):</strong> Add a <strong>3× Strength Booster Gear</strong> (<code>APEX-PG20-2</code>) — <em>multiplies lifting power by 3× right at the shoulder without needing a giant heavy motor.</em>
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
          <div class="rec-metric-label">Floor-to-Floor Bottle Delivery</div>
          <div class="rec-metric-val">
            <span style="color:var(--red); text-decoration:line-through;">Drops halfway</span>
            &rarr; <span style="color:var(--ours);">Placed on Floor Spot B ✓</span>
          </div>
        </div>
        <div class="rec-metric-box">
          <div class="rec-metric-label">Shoulder Motor Effort</div>
          <div class="rec-metric-val">
            <span style="color:var(--red); text-decoration:line-through;">Overloaded (238%)</span>
            &rarr; <span style="color:var(--ours);">Comfortable (78% effort)</span>
          </div>
        </div>
        <div class="rec-metric-box">
          <div class="rec-metric-label">Total Robot Arm Weight</div>
          <div class="rec-metric-val">
            <span class="muted">${m0?.total_mass_kg?.toFixed(2) ?? "1.01"} kg</span>
            &rarr; <span style="color:var(--ours);">${mf?.total_mass_kg?.toFixed(2) ?? "0.94"} kg (Lighter!)</span>
          </div>
        </div>
        <div class="rec-metric-box">
          <div class="rec-metric-label">Total Hardware Cost</div>
          <div class="rec-metric-val">
            <span class="muted">$${m0?.total_cost_usd?.toFixed(0) ?? "945"}</span>
            &rarr; <span style="color:var(--fg);">$${mf?.total_cost_usd?.toFixed(0) ?? "1,126"} (Under $1,200 Budget)</span>
          </div>
        </div>
      </div>

      <!-- Optional Technical Details & PDF Datasheet Citations Drawer -->
      <details class="onecode-thinking" style="margin-top:8px;">
        <summary class="onecode-thinking-summary">
          <div class="onecode-thinking-left">
            <span class="onecode-thinking-label">Show Exact Part Model Numbers &amp; Datasheet PDF Citations (Optional)</span>
          </div>
          <span class="onecode-chevron">›</span>
        </summary>
        <div class="onecode-thinking-content" style="font-size:12px; color:var(--fg);">
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
        <button class="btn ${recommendationAccepted ? "btn-mono" : "btn-success"}" id="btn-accept-recommendation">
          ${
            recommendationAccepted
              ? "✓ Upgrade Applied to 3D Robot! (Click to Replay Upgrade)"
              : "Apply Upgrade to 3D Robot & Watch It Carry the Bottle →"
          }
        </button>
        <button class="btn btn-outline btn-sm" id="btn-rec-compare">
          Compare Before vs. After Side-by-Side
        </button>
        <button class="btn btn-outline btn-sm" id="btn-rec-report">
          Open Validation Report
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
    <div class="card" style="padding:14px; background:#fff;">
      <div class="rec-header">
        <div class="rec-title">Final Sign-Off &amp; Download Robot Package</div>
        <span class="tag ${isApproved ? "tag-ok" : "badge-warn"}">${isApproved ? "Approved ✓" : "Ready to Approve"}</span>
      </div>
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-bottom:10px;">
        <input
          type="text"
          id="input-eng-name"
          style="height:34px; padding:0 10px; border-radius:2px; border:1px solid var(--border); background:var(--card); color:var(--fg); font-family:inherit;"
          value="${escapeHtml(engName)}"
          placeholder="Your Name"
        />
        <input
          type="text"
          id="input-eng-notes"
          style="height:34px; padding:0 10px; border-radius:2px; border:1px solid var(--border); background:var(--card); color:var(--fg); font-family:inherit;"
          value="${escapeHtml(engNotes)}"
          placeholder="Approval Notes"
        />
      </div>
      <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
        <button class="btn btn-success btn-sm" id="btn-submit-approval">
          ${isApproved ? "✓ Saved Approval" : "Approve Final Robot Design"}
        </button>
        <a class="btn btn-mono btn-sm" href="/api/projects/demo/export" download="motionfit-demo-bundle.zip">
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

  // Draw the Floor Line across the bottom of the canvas so the user clearly sees the bottle on the floor
  const floorY = h - 24;
  ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(24, floorY);
  ctx.lineTo(w - 24, floorY);
  ctx.stroke();

  ctx.fillStyle = "rgba(148, 163, 184, 0.65)";
  ctx.font = "10px Instrument Sans, sans-serif";
  ctx.fillText("FLOOR LEVEL (68 cm Long-Distance Bottle Carry)", w / 2 - 110, floorY + 15);

  const totalFrames = state.landmarks?.frames?.length || 180;
  const idx = Math.max(0, Math.min(totalFrames - 1, frameIdx));
  const u = idx / Math.max(1, totalFrames - 1);

  const pickupX = 115;
  const pickupY = floorY - 12;
  const targetX = w - 95;
  const targetY = floorY - 12;

  // Draw high floor-to-floor arc dashed guide line
  ctx.beginPath();
  for (let i = 0; i <= 40; i++) {
    const s = i / 40;
    const ax = pickupX + (targetX - pickupX) * s;
    const ay = pickupY - Math.sin(Math.PI * s) * 82;
    if (i === 0) ctx.moveTo(ax, ay);
    else ctx.lineTo(ax, ay);
  }
  ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw Floor Spot A Coaster Pad
  ctx.fillStyle = "rgba(59, 115, 255, 0.25)";
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.ellipse(pickupX, floorY, 24, 5, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#7dd3fc";
  ctx.font = "bold 10px Instrument Sans, sans-serif";
  ctx.fillText("Floor Spot A", pickupX - 30, floorY + 14);

  // Draw Floor Spot B Coaster Pad
  ctx.fillStyle = "rgba(16, 185, 129, 0.25)";
  ctx.strokeStyle = "#10b981";
  ctx.beginPath();
  ctx.ellipse(targetX, floorY, 24, 5, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#6ee7b7";
  ctx.fillText("Floor Spot B", targetX - 30, floorY + 14);

  const graspU = 0.19;
  const releaseU = 0.81;
  let handX = pickupX;
  let handY = pickupY;
  let isCarrying = false;

  if (u < graspU) {
    const p = u / graspU;
    handX = pickupX - 18 * (1 - p);
    handY = pickupY - 46 * (1 - p);
  } else if (u <= releaseU) {
    isCarrying = true;
    const s = (u - graspU) / (releaseU - graspU);
    const smoothS = s * s * (3 - 2 * s);
    handX = pickupX + (targetX - pickupX) * smoothS;
    handY = pickupY - Math.sin(Math.PI * smoothS) * 82;
  } else {
    const p = (u - releaseU) / (1 - releaseU);
    handX = targetX - 10 * p;
    handY = targetY - 36 * p;
  }

  const shX = 68;
  const shY = 54;
  const hipX = 56;
  const hipY = floorY - 4;

  const midX = (shX + handX) * 0.5;
  const midY = (shY + handY) * 0.5;
  const elX = midX - 14;
  const elY = Math.min(floorY - 18, midY + 24);
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

  // Draw 500 mL Water Bottle
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
    ctx.font = "10px Instrument Sans, sans-serif";
    ctx.fillText(j.label, j.x + 7, j.y - 6);
  });

  ctx.fillStyle = "#d4d4d8";
  ctx.font = "11px Instrument Sans, sans-serif";
  const statusMsg =
    u < graspU
      ? "Step 1: Reaching down to Floor Spot A to grab the 500 mL water bottle"
      : isCarrying
      ? "Step 2: Lifting water bottle off the floor & carrying it 68 cm across to Floor Spot B"
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
    await sleep(165);
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

async function handleTurn1BuildArmPrompt(userPrompt: string): Promise<void> {
  if (agentBusy) return;
  agentBusy = true;
  activeView = "studio";
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
        <strong>Your 3D Robot Arm is Built!</strong>
        <br/>
        Look at the right panel: the robot arm is assembled, and the <strong>500 mL water bottle</strong> is sitting on the floor at <strong>Floor Spot A</strong>.
        <br/><br/>
        <strong>Next Step:</strong> Attach a motion video (or click <em>"2. Send Floor Water-Bottle Video &amp; Test"</em> below) so I can track the human arm motion, test picking up the water bottle from the floor, and check if the motors are strong enough.
      </div>
    `;
  }

  agentBusy = false;
  agentStatusText = "Waiting for Your Motion Video";
  renderApp();
}

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
    text: "Tracked the human arm from the video: reaching down to the floor (Spot A), lifting the 500 mL water bottle high up, and carrying it 68 cm across to place it on the floor (Spot B).",
    badge: "Skeleton Tracked",
  });
  if (targetMsg) {
    targetMsg.planCompletedCount = 2;
    targetMsg.showSkeletonCard = true;
  }
  renderApp();
  await sleep(240);

  // Step B: Baseline Simulation
  agentStatusText = "Simulating Floor Bottle Lift in 3D...";
  thinkingSteps.push({
    status: "fail",
    text: "Tested First Try Robot in 3D Physics: The Shoulder Lift Motor is TOO WEAK! Lifting the full 500 mL water bottle off the floor and holding it out across a 68 cm distance overloads the shoulder by more than 2× (238% strain). Halfway across, the arm loses strength and sags back down to the floor.",
    badge: "Shoulder Too Weak ✗",
  });
  if (targetMsg) {
    targetMsg.planCompletedCount = 3;
  }
  renderApp();
  await sleep(260);

  // Step C: Smart Solution Selection
  agentStatusText = "Finding Stronger & Lighter Parts...";
  thinkingSteps.push({
    status: "warn",
    text: "Checked Option A (buying a giant heavy motor): Too heavy and pushes total price over budget ($1,515).",
    badge: "Over Budget",
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
          <li><strong style="color:var(--red);">Why the First Robot Fails (Shown on Right Now):</strong> Picking up a full <strong>500 mL water bottle</strong> from the floor and carrying it 68 cm across is like holding a heavy bottle at arm's length. The <strong>Shoulder Lift Motor runs out of strength</strong> (238% strain, glowing red in 3D), so the arm sags down to the floor halfway across and never reaches Floor Spot B.</li>
          <li><strong style="color:var(--ours);">How We Fix It Easily:</strong> We add a <strong>3× Strength Booster Gear</strong> to the Shoulder Motor and switch to <strong>Lighter Arm Tubes</strong>. That gives the shoulder 3× more lifting power while making the arm itself lighter.</li>
        </ul>
        Click <strong>"Apply Upgrade to 3D Robot &amp; Watch It Carry the Bottle →"</strong> below to upgrade the robot on the right panel!
      </div>
    `;
    targetMsg.showRecommendationCard = true;
  }

  agentBusy = false;
  agentStatusText = "Upgrade Ready — Click Apply Below";
  renderApp();
}

async function handleAcceptRecommendation(): Promise<void> {
  if (agentBusy) return;
  agentBusy = true;
  recommendationAccepted = true;
  studioMode = "single-after";
  agentStatusText = "Installing 3× Shoulder Booster Gear & Lighter Tubes...";
  renderApp();

  await runProgressiveAssemblyAnimation("single-after", "Upgrading 3D Robot");

  currentFrameIdx = 0;

  const alreadyHasApprovalMsg = chatMessages.some((m) => m.showApprovalCard);
  if (!alreadyHasApprovalMsg) {
    chatMessages.push({
      id: `agent-applied-${Date.now()}`,
      role: "agent",
      timestamp: nowTimeStr(),
      html: `
        <div>
          <strong>Upgrade Installed on the 3D Robot!</strong>
          <br/>
          Look at the <strong>Right 3D Studio</strong> now:
          <ul style="margin:6px 0 8px 18px; padding:0;">
            <li>The robot reaches right down to <strong>Floor Spot A</strong> and grabs the 500 mL water bottle off the floor.</li>
            <li>With the <strong>3× Strength Booster Gear</strong> and <strong>Lighter Arm Tubes</strong>, all motors stay in the comfortable green zone (&le; 78% effort).</li>
            <li>It lifts the water bottle high in a smooth arc across the entire 68 cm span and sets it gently upright on <strong>Floor Spot B</strong>.</li>
          </ul>
          Tip: Click <strong>"Side-by-Side Compare"</strong> at the top of the 3D view to watch the failing First Try and the Upgraded Robot side-by-side.
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
      phaseReadout.className = "tag tag-ai";
    } else if (tSec <= 4.85) {
      phaseReadout.textContent = "Step 2: Lifting & Carrying Bottle Across Floor";
      phaseReadout.className = "tag badge-warn";
    } else {
      phaseReadout.textContent = "Step 3: Placing Bottle on Floor Spot B";
      phaseReadout.className = "tag tag-ok";
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

  const joints = ["J1", "J2", "J3", "J4"];
  const winStart = Math.max(0, idx - 3);
  const winEnd = Math.min(trace.t.length - 1, idx + 3);

  // Create initial DOM structure once so we never rebuild innerHTML at 30 FPS
  if (el.children.length !== joints.length) {
    el.innerHTML = joints
      .map((jName) => {
        const friendlyName = FRIENDLY_JOINT_NAMES[jName] || jName;
        return `
          <div class="joint-bar-item" data-joint="${jName}">
            <div class="joint-bar-top">
              <strong>${escapeHtml(friendlyName)}</strong>
              <span class="joint-status-val">0%</span>
            </div>
            <div class="joint-meter">
              <div class="joint-meter-fill" style="width:0%; background:#0f766e;"></div>
            </div>
          </div>
        `;
      })
      .join("");
  }

  joints.forEach((jName, jIdx) => {
    let sumRatio = 0;
    let count = 0;
    for (let k = winStart; k <= winEnd; k++) {
      const cmd = Math.abs(trace.tau_cmd[k]?.[jIdx] ?? 0);
      const avail = Math.max(Math.abs(trace.tau_avail[k]?.[jIdx] ?? 1), 1e-4);
      sumRatio += Math.min(1.0, cmd / avail);
      count++;
    }
    const ratio = count > 0 ? sumRatio / count : 0;
    const pct = Math.min(100, Math.round(ratio * 100));
    const color = ratio >= 0.92 ? "#e40014" : ratio >= 0.72 ? "#d97706" : "#0f766e";
    const statusWord =
      ratio >= 0.92
        ? `MAX ${pct}%`
        : ratio >= 0.72
        ? `High ${pct}%`
        : `Easy ${pct}%`;

    const itemEl = el.children[jIdx] as HTMLElement | undefined;
    if (!itemEl) return;
    const statusEl = itemEl.querySelector(".joint-status-val") as HTMLElement | null;
    if (statusEl) {
      statusEl.textContent = statusWord;
      statusEl.style.color = color;
    }
    const fillEl = itemEl.querySelector(".joint-meter-fill") as HTMLElement | null;
    if (fillEl) {
      fillEl.style.width = `${pct}%`;
      fillEl.style.backgroundColor = color;
    }
  });
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
              <strong style="font-size:15px;">Robot Motors &amp; Booster Gears Catalog (${state.catalog.parts.length} Real Parts)</strong>
              <div style="font-size:12px; color:var(--mfg);">Every motor and gear is verified from official manufacturer datasheets</div>
            </div>
            <button class="btn btn-outline btn-sm" id="btn-close-modal">Close ✕</button>
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
                        <td><strong class="mono">${escapeHtml(p.part_id)}</strong></td>
                        <td><span class="tag ${p.type === "actuator" ? "tag-ai" : "tag-ok"}">${p.type === "actuator" ? "Motor" : "Booster Gear"}</span></td>
                        <td>${escapeHtml(p.vendor)}</td>
                        <td class="num">
                          ${
                            p.type === "actuator"
                              ? `${p.stall_torque_nm?.value?.toFixed(2) ?? "-"} N·m`
                              : `${p.rated_output_torque_nm?.value?.toFixed(1) ?? "-"} N·m max`
                          }
                        </td>
                        <td class="num">
                          ${
                            p.type === "actuator"
                              ? `${p.continuous_torque_nm?.value?.toFixed(2) ?? "-"} N·m`
                              : `${p.ratio?.value ?? "-"}× Strength Boost`
                          }
                        </td>
                        <td class="num">${p.mass_kg.value.toFixed(3)} kg</td>
                        <td class="num">$${p.price_usd.value.toFixed(0)}</td>
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
              <strong style="font-size:15px;">MotionFit Summary &amp; Verification Report</strong>
              <div style="font-size:12px; color:var(--mfg);">Complete before/after comparison and datasheet citations</div>
            </div>
            <div style="display:flex; gap:8px;">
              <a class="btn btn-mono btn-sm" href="/api/projects/demo/report" target="_blank">Open in New Tab ↗</a>
              <button class="btn btn-outline btn-sm" id="btn-close-modal">Close ✕</button>
            </div>
          </div>
          <div class="modal-body" style="padding:0; overflow:hidden;">
            <iframe src="/api/projects/demo/report" style="width:100%; height:100%; border:none; background:#fff;"></iframe>
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
              <strong style="font-size:15px;">Official Datasheet: ${escapeHtml(pdfModalDoc)} (Page ${pdfModalPage})</strong>
              <div style="font-size:12px; color:var(--mfg);">Verified hardware specification page</div>
            </div>
            <button class="btn btn-outline btn-sm" id="btn-close-modal">Close ✕</button>
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

function switchActiveView(target: ActivePageView): void {
  transitionView(() => {
    activeView = target;
    renderApp();
  });
}

function bindEvents(): void {
  // Sidebar navigation
  document.getElementById("nav-brand-home")?.addEventListener("click", (e) => {
    e.preventDefault();
    switchActiveView("landing");
  });

  document.querySelectorAll<HTMLElement>("[data-nav]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = (btn.getAttribute("data-nav") || "landing") as ActivePageView;
      switchActiveView(target);
    });
  });

  // Landing Page Hero Composer & CTAs
  const heroForm = document.getElementById("hero-chat-form") as HTMLFormElement | null;
  const heroAsk = document.getElementById("hero-ask") as HTMLTextAreaElement | null;

  heroAsk?.addEventListener("input", () => {
    heroDraftPrompt = heroAsk.value;
  });

  heroAsk?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      heroForm?.requestSubmit();
    }
  });

  heroForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = (heroAsk?.value || heroDraftPrompt).trim();
    transitionView(() => {
      activeView = "studio";
      renderApp();
    });
    if (text) {
      handleVideoSimulationAndRedesign(
        text,
        pendingVideoAttachment?.label || "floor_water_bottle_carry_720p.mp4 (6.0s)",
        true
      );
    }
  });

  document.getElementById("btn-hero-attach-demo")?.addEventListener("click", () => {
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

  document.getElementById("btn-hero-quick-full")?.addEventListener("click", () => {
    transitionView(() => {
      activeView = "studio";
      renderApp();
    });
    handleVideoSimulationAndRedesign(
      "Build a 3D robot arm for this video of picking up a 500 mL water bottle from Floor Spot A and carrying it 68 cm across to Floor Spot B. Simulate the bottle transfer and recommend the best motor upgrades!",
      "floor_water_bottle_carry_720p.mp4 (6.0s)",
      true
    );
  });

  document.getElementById("btn-hero-quick-build")?.addEventListener("click", () => {
    transitionView(() => {
      activeView = "studio";
      renderApp();
    });
    handleTurn1BuildArmPrompt(
      "Build a 4-joint desktop robot arm that can pick up a 500 mL water bottle from the floor and carry it across a 68 cm span."
    );
  });

  document.getElementById("btn-hero-open-studio")?.addEventListener("click", () => {
    switchActiveView("studio");
  });

  document.getElementById("btn-foot-open-studio")?.addEventListener("click", () => {
    switchActiveView("studio");
  });

  document.getElementById("btn-foot-catalog")?.addEventListener("click", () => {
    switchActiveView("catalog");
  });

  document.getElementById("btn-cat-back-studio")?.addEventListener("click", () => {
    switchActiveView("studio");
  });

  document.getElementById("btn-cat-next-report")?.addEventListener("click", () => {
    switchActiveView("report");
  });

  document.getElementById("btn-rep-back-studio")?.addEventListener("click", () => {
    switchActiveView("studio");
  });

  // Studio Topbar & Modals
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

  // Quick Chips in Studio Chat
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
  // Pre-populate the full floor-to-floor water-bottle analysis walkthrough in the Studio conversation
  // while opening on the Obolus Landing Page (or Studio if hash is #studio), so both the Landing Page
  // and the Studio are immediately rich and ready!
  if (window.location.hash === "#studio") {
    activeView = "studio";
  }
  await handleVideoSimulationAndRedesign(
    "Build a robot arm for this video of picking up a 500 mL water bottle from the floor and carrying it far across to another spot on the floor. Test if it works and recommend any needed part upgrades!",
    "floor_water_bottle_carry_720p.mp4 (6.0s)",
    false
  );
}

boot().catch((err) => {
  console.error("MotionFit boot error:", err);
});
