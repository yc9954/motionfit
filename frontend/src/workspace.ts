import { RobotArmViewport, type DesignSpec, type TraceData } from "./viewer";
import { extractLandmarksFromVideoFile, drawPoseOverlay } from "./pose";

type Message = { role: "user" | "agent"; text: string; at: string };
type Project = {
  project_id: string; design: DesignSpec; catalog: { parts: any[]; confirmed?: boolean; [key: string]: any };
  task?: any; target?: any; landmarks?: any; loop_result?: any; workflow?: any; accepted?: boolean;
  approval?: { approved: boolean }; has_report?: boolean; final_design?: DesignSpec;
};
let project: Project | null = null;
let messages: Message[] = [];
let busy = false;
let sourceFile: File | null = null;
let sourceUrl: string | null = null;
let overlayFrames: any[] = [];
let candidateVisible = false;
let accepted = false;
let playing = false;
let playhead = 0;
let lastTime = 0;
let speed = 1;
let beforeViewer: RobotArmViewport | null = null;
let afterViewer: RobotArmViewport | null = null;
let resultIdentity = "";
let followChat = true;
let runId = "";
let progress = "Describe your task and attach a motion video.";
let activeEventSource: EventSource | null = null;
let alive = true;
let activeAbort: AbortController | null = null;

const app = document.querySelector<HTMLDivElement>("#app")!;
const escape = (v: unknown) => String(v ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const num = (v: unknown, digits = 2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
const apiPath = (tail = "") => `/api/projects/${encodeURIComponent(project!.project_id)}${tail}`;

async function request<T = any>(path: string, method = "GET", data?: any): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: data instanceof FormData ? undefined : { "Content-Type": "application/json" },
    body: data === undefined ? undefined : data instanceof FormData ? data : JSON.stringify(data),
  });
  if (!response.ok) {
    let detail: any;
    try { detail = (await response.json()).detail; } catch { detail = response.statusText; }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail || `HTTP ${response.status}`));
  }
  return response.json();
}
function historyKey() { return `motionfit.chat.${project?.project_id}`; }
function persist() {
  if (!project) return;
  try { localStorage.setItem("motionfit.project", project.project_id); localStorage.setItem(historyKey(), JSON.stringify(messages)); } catch { /* Storage may be disabled; the project remains on the server. */ }
}
function say(role: Message["role"], text: string) {
  messages.push({ role, text, at: new Date().toISOString() });
  persist(); renderMessages();
}
function renderMessages() {
  const list = $("chat-stream");
  const wasNearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 90;
  list.innerHTML = messages.length ? messages.map(m => `<article class="message ${m.role}"><div class="message-meta">${m.role === "user" ? "You" : "MotionFit"}<time>${new Date(m.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time></div><div class="message-text">${escape(m.text)}</div></article>`).join("") : `<div class="welcome"><span class="eyebrow">YOUR MOTION. A VERIFIED DESIGN.</span><h1>Build for the way<br>you need to move.</h1><p>Describe the task, specify the load, and attach your motion. We simulate catalog parts against your recorded path and show the measured result here.</p><p class="small">This workspace supports a 4-joint arm. The simulation checks joint loads and tracking. Grasping, collisions and manufacturing fit require separate validation.</p></div>`;
  if (followChat || wasNearBottom) list.scrollTop = list.scrollHeight;
}
function setProgress(text: string) { progress = text; $("progress-text").textContent = text; $("busy-dot").classList.toggle("active", busy); }
function setBusy(value: boolean) {
  busy = value;
  ["send", "run", "new-project", "confirm-catalog", "upload-catalog", "accept", "approve", "input-video-file", "import-landmarks", "design-file", "catalog-file", "extract-pdfs"].forEach(id => { const el = $(id) as HTMLButtonElement; if (el) el.disabled = value; });
  $("prompt").setAttribute("aria-busy", String(value));
  refreshActions();
}
function failure(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  setProgress("Action failed. Review the error in the conversation.");
  say("agent", `Could not complete this action: ${message}\nNo replacement motion or successful result was generated.`);
}
function runAction(action: () => Promise<void>) {
  if (busy) return;
  setBusy(true);
  action().catch(failure).finally(() => setBusy(false));
}
async function refreshProject() {
  project = await request<Project>(apiPath());
  accepted = Boolean(project.accepted || project.workflow?.accepted);
  persist(); refreshActions();
}
async function ensureProject() {
  if (project) return;
  project = await request<Project>("/api/projects", "POST", { use_demo: false });
  persist(); refreshActions();
}
function hasConfirmedCatalog() { return Boolean(project?.catalog?.parts?.length && project.catalog.parts.every((p: any) => p.user_confirmed)); }
function refreshActions() {
  $("project-name").textContent = project ? project.project_id : "New workspace";
  $("catalog-label").textContent = project ? `${project.catalog?.parts?.length ?? 0} catalog parts · ${hasConfirmedCatalog() ? "reviewed" : "review required"}` : "Review parts";
  const hasResult = Boolean(project?.loop_result);
  $("run").toggleAttribute("disabled", busy || !project?.target || !hasConfirmedCatalog());
  $("accept").toggleAttribute("disabled", busy || !project?.loop_result?.passed || accepted);
  $("approve").toggleAttribute("disabled", busy || !accepted || Boolean(project?.approval?.approved));
  $("accept").textContent = accepted ? "Candidate applied" : "Apply verified candidate";
  $("approve").textContent = project?.approval?.approved ? "Approval recorded" : "Sign off";
  $("compare").toggleAttribute("disabled", !hasResult);
  $("report").toggleAttribute("disabled", !project?.has_report);
  const link = $<HTMLAnchorElement>("export");
  link.href = project && project.approval?.approved ? apiPath("/export.zip") : "#";
  link.setAttribute("aria-disabled", String(!project?.approval?.approved));
  $("result-state").textContent = !hasResult ? "Awaiting verification" : `${project!.loop_result.passed ? "Passes numerical checks" : "No passing design found"}${accepted ? " · Applied" : " · Candidate"}`;
  $("result-state").className = `pill ${hasResult ? project!.loop_result.passed ? "pass" : "fail" : ""}`;
  $("task-title").textContent = project?.task?.summary || project?.task?.task_summary || "Your robot workspace";
  $("task-facts").textContent = project?.task ? `Payload ${num(project.design.payload_kg, 3)} kg · ${project.design.topology || "4-joint arm"}` : "A measured trajectory will appear after you upload and verify a motion.";
}

app.innerHTML = `
<header class="topbar"><a class="brand" href="#" aria-label="MotionFit home"><span class="mascot" aria-hidden="true"><svg viewBox="0 0 90 70"><defs><linearGradient id="m" x2="1" y2="1"><stop stop-color="#97c8ed"/><stop offset=".5" stop-color="#aaa2ed"/><stop offset="1" stop-color="#526ac0"/></linearGradient></defs><path d="M13 52C1 32 18 10 39 8S84 11 82 36 61 65 38 62 18 63 13 52" fill="url(#m)"/><ellipse cx="35" cy="34" rx="4" ry="7" fill="#202b40"/><ellipse cx="54" cy="34" rx="4" ry="7" fill="#202b40"/></svg></span>MotionFit</a><span class="project-name" id="project-name">New workspace</span><div class="top-actions"><button id="catalog-label">Review parts</button><button id="new-project">New project</button></div></header>
<main class="workspace"><section class="conversation" aria-label="Design conversation"><div class="chat-stream" id="chat-stream" aria-live="polite"></div><div class="progress" role="status"><span id="busy-dot"></span><span id="progress-text"></span></div><form id="chat-form" class="composer"><textarea id="prompt" rows="3" placeholder="What should your robot do? Include the load, budget and reach you need." aria-label="Task description"></textarea><details id="constraints-panel" open><summary>Load & design constraints <span>Specify what the video cannot measure</span></summary><div class="constraint-grid"><label>Payload (kg)<input id="payload" type="number" min="0" step="0.01" placeholder="Required"/></label><label>Budget (USD)<input id="budget" type="number" min="1" step="1" placeholder="Optional"/></label><label>Max arm mass (kg)<input id="mass" type="number" min="0.01" step="0.01" placeholder="Optional"/></label><label>Minimum reach (m)<input id="reach" type="number" min="0.1" max="0.83" step="0.01" placeholder="Optional"/></label><label>Tool orientation<select id="orientation"><option value="">Infer from task</option><option value="free">Follow recorded orientation</option><option value="keep_level">Keep level</option><option value="keep_upright">Keep upright</option></select></label><label>Tracked arm<select id="arm-side"><option value="right">Right arm</option><option value="left">Left arm</option></select></label></div></details><div class="attachment" id="attachment" hidden></div><div class="composer-actions"><label class="file-button">＋ Motion video<input id="input-video-file" type="file" accept="video/*"/></label><label class="file-button secondary-file">Pose JSON<input id="import-landmarks" type="file" accept=".json,application/json"/></label><button class="primary" id="send" type="submit">Send & design <span>↗</span></button></div><p class="composer-note">Pose extraction runs locally. Detection errors stop the analysis. No demonstration trajectory is substituted.</p></form></section>
<section class="studio" aria-label="Robot verification studio"><div class="studio-heading"><div><span class="eyebrow">DESIGN STUDIO</span><h2 id="task-title">Your robot workspace</h2><p id="task-facts"></p></div><span class="pill" id="result-state">Awaiting verification</span></div><div class="studio-toolbar"><button id="run">Verify motion</button><button id="compare" disabled>Preview candidate / compare</button><button id="axes" aria-pressed="true">Joint axes</button><button id="ghost" aria-pressed="true">Target overlay</button><label class="file-button">Import arm JSON<input id="design-file" type="file" accept=".json,application/json"/></label></div><div class="view-grid" id="view-grid"><div class="view-card"><span class="view-label" id="before-label">Current design</span><div class="viewport" id="viewport-before"></div><div class="empty-view" id="empty-view"><span class="empty-glyph">◎</span><h3>Designed around your motion</h3><p>Describe a task to build the initial arm.<br>Upload a motion and review its measured verification.</p></div><div class="joint-strip" id="live-strip-before"></div></div><div class="view-card" id="after-card" hidden><span class="view-label">Verified candidate · not applied until accepted</span><div class="viewport" id="viewport-after"></div><div class="joint-strip" id="live-strip-after"></div></div></div><div class="timeline"><button id="play" disabled>▶ Play</button><input id="timeline-slider" type="range" min="0" max="0" value="0" aria-label="Simulation time" disabled/><span id="time-label">0.00 s</span><select id="speed" aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></div><div class="metrics" id="metrics"></div><div class="source-panel"><div class="source-title"><strong>Source motion & detected pose</strong><span id="source-label">No video attached</span></div><div class="video-wrap" id="video-wrap" hidden><video id="source-video" controls muted playsinline></video><canvas id="skeleton-canvas"></canvas></div><div class="source-empty" id="source-empty">Only detected keypoints from your recording appear here. Upload a video or captured pose JSON to begin.</div></div><div class="result-actions"><button class="primary" id="accept" disabled>Apply verified candidate</button><button id="approve" disabled>Sign off</button><button id="report" disabled>Report</button><a id="export" href="#" aria-disabled="true">Download MJCF + JSON bundle</a></div><p class="model-note">Parametric part envelopes, sized from catalog dimensions. Numerical arm-load verification does not certify mechanical fit, thermal duty, collision clearance or successful grasping.</p></section></main>
<dialog id="catalog-dialog"><div class="dialog-header"><h2>Review your catalog</h2><button data-close="catalog-dialog" aria-label="Close catalog">✕</button></div><p>Bundled PDFs are project-generated reference sheets, not manufacturer originals. Review dimensions, units, prices and sources. Derived continuous torque is an estimate. Confirm only the data you intend to simulate.</p><div id="catalog-summary"></div><label class="file-button">Import catalog JSON<input type="file" id="catalog-file" accept=".json,application/json"/></label><textarea id="catalog-json" spellcheck="false" aria-label="Editable part catalog JSON"></textarea><details><summary>Extract your own source PDFs</summary><p>Extraction requires the configured model. Failures are reported; no catalog is substituted.</p><input id="datasheet-files" type="file" accept=".pdf,.csv" multiple/><button id="upload-catalog">Upload source files</button><button id="extract-pdfs">Extract uploaded PDFs</button></details><button class="primary" id="confirm-catalog">Save reviewed catalog</button></dialog>
<dialog id="approval-dialog"><div class="dialog-header"><h2>Engineering sign-off</h2><button data-close="approval-dialog">✕</button></div><p>Sign-off applies to the current verified input version. Check the report and model scope before exporting.</p><label>Your name<input id="engineer" required/></label><label>Review notes<textarea id="approval-notes" rows="3"></textarea></label><button class="primary" id="save-approval">Record approval</button></dialog>
<dialog id="report-dialog"><div class="dialog-header"><h2>Verification report</h2><a id="report-tab" target="_blank" rel="noopener">Open in new tab ↗</a><button data-close="report-dialog">✕</button></div><iframe id="report-frame" title="Engineering verification report"></iframe></dialog>`;

function constraints() {
  const values: Record<string, number> = {};
  for (const [id, key] of [["payload", "payload_kg"], ["budget", "max_cost_usd"], ["mass", "max_total_mass_kg"], ["reach", "min_reach_m"]]) {
    const raw = $<HTMLInputElement>(id).value.trim();
    if (raw !== "") { const value = Number(raw); if (!Number.isFinite(value) || value < 0) throw new Error(`Invalid ${key}.`); values[key] = value; }
  }
  return values;
}
async function submitTask() {
  const text = $<HTMLTextAreaElement>("prompt").value.trim();
  if (!text && !sourceFile) throw new Error("Describe the task or attach a motion video.");
  await ensureProject();
  const prompt = text || project?.task?.prompt || "Verify my recorded arm motion under the entered constraints.";
  const selectedFile = sourceFile;
  say("user", `${prompt}${selectedFile ? `\nAttached video: ${selectedFile.name}` : ""}`);
  $<HTMLTextAreaElement>("prompt").value = "";
  setProgress("Interpreting your task and constraints…");
  const previousLandmarks = project?.landmarks;
  const taskResult = await request(apiPath("/task"), "POST", { prompt, constraints: constraints(), orientation_constraint: $<HTMLSelectElement>("orientation").value || undefined });
  accepted = false; candidateVisible = false; resultIdentity = "";
  await refreshProject();
  const task = taskResult.task || project?.task || {};
  say("agent", `${task.summary || task.task_summary || "Task recorded."}\nInput interpretation: ${task.interpretation_source || "user constraints"}.${task.interpretation_notice ? `\n${task.interpretation_notice}` : ""}${task.assumptions?.length ? `\nAssumptions to review:\n${task.assumptions.join("\n")}` : ""}`);
  if (task.orientation_constraint) $<HTMLSelectElement>("orientation").value = task.orientation_constraint;
  rebuildViewers();
  if (selectedFile) {
    setProgress("Detecting actual arm landmarks in your recording…");
    activeAbort = new AbortController();
    const payload = await extractLandmarksFromVideoFile(selectedFile, {
      videoEl: $<HTMLVideoElement>("source-video"), overlayCanvas: $<HTMLCanvasElement>("skeleton-canvas"),
      armSide: $<HTMLSelectElement>("arm-side").value as "right" | "left",
      signal: activeAbort.signal,
      onProgress: (pct: number, status: string) => setProgress(`Video ${Math.round(pct)}% · ${status}`),
    });
    overlayFrames = payload.overlayFrames || [];
    await request(apiPath("/motion"), "POST", { landmarks: payload, orientation_constraint: $<HTMLSelectElement>("orientation").value || project?.task?.orientation_constraint || "free" });
    await refreshProject();
    sourceFile = null; $("attachment").hidden = true;
    $<HTMLVideoElement>("source-video").currentTime = 0;
    say("agent", `Detected ${payload.frames.length} motion frames from ${selectedFile.name}. The mapped robot target is ${num(project!.target?.duration_s, 2)} s long. It is derived from your recording; load and budget come from the task inputs.`);
  } else if (previousLandmarks?.frames?.length) {
    await request(apiPath("/motion"), "POST", { landmarks: previousLandmarks, orientation_constraint: $<HTMLSelectElement>("orientation").value || project?.task?.orientation_constraint || "free" });
    await refreshProject();
    say("agent", "Reused this project's recorded motion with the updated task constraints.");
  }
  const missing = taskResult.missing_fields || task.missing_fields || [];
  if (missing.length) {
    $("constraints-panel").setAttribute("open", "");
    say("agent", `Before verification, specify: ${missing.join(", ")}. A video cannot establish payload mass. Enter the missing values and send the task again.`);
    setProgress("Waiting for required task constraints."); return;
  }
  if (!project?.target) { say("agent", "Initial arm created. Attach a motion video or import captured pose JSON to verify this task. No motion has been inferred or simulated yet."); setProgress("Waiting for your motion recording."); return; }
  if (!hasConfirmedCatalog()) { say("agent", "Motion is ready. Review the part catalog and save the values you intend to use, then click Verify motion. Bundled sheets and derived ratings are explicitly marked assumptions."); setProgress("Catalog review required before simulation."); return; }
  await startRun();
}
function eventText(e: any): string | null {
  const kind = e.event || e.type;
  switch (kind) {
    case "loop_started": return "Verification started on the current input version.";
    case "simulating": return `Simulating candidate ${Number(e.iteration) + 1} with MuJoCo…`;
    case "iteration_simulated": return `Candidate ${Number(e.iteration) + 1}: ${e.passed ? "passes" : "fails"} numerical checks · path error ${num(e.ee_rmse_mm)} mm · mass ${num(e.total_mass_kg, 3)} kg · cost $${num(e.total_cost_usd)}.`;
    case "argon_thinking": return "Evaluating the measured violations against catalog candidates…";
    case "argon_proposed": return `Proposed ${e.changes?.length ?? 0} changes (${e.used_live_api ? e.model || "model" : "deterministic catalog search"}). ${e.diagnosis || ""}`;
    case "proposal_validation_failed": return `Rejected proposal: ${(e.errors || []).join("; ")}`;
    case "proposal_validated": return "Catalog references and design constraints validated. Re-simulating changes.";
    case "loop_aborted": return `Stopped: ${e.reason || "proposal rejected"}. ${(e.errors || []).join("; ")}`;
    case "no_feasible_proposal": return e.reason || "No further candidate satisfies the proposal constraints.";
    default: return null;
  }
}
async function startRun() {
  if (!project?.target) throw new Error("Upload a valid recorded motion before verification.");
  if (!hasConfirmedCatalog()) throw new Error("Review and confirm the catalog before verification.");
  setProgress("Starting numerical verification…");
  const start = await request(apiPath("/run"), "POST", { max_iterations: 5, enable_post_pass_optimization: true, include_naive_baseline: false });
  runId = start.run_id;
  if (!runId) throw new Error("The server did not return a run identifier.");
  try { localStorage.setItem(`motionfit.run.${project.project_id}`, runId); } catch { /* optional */ }
  await watchRun(runId);
}
async function watchRun(id: string) {
  const seen = new Set<string>();
  const consume = (event: any) => {
    const key = JSON.stringify(event);
    if (seen.has(key)) return; seen.add(key);
    const text = eventText(event);
    if (text) { setProgress(text); say("agent", text); }
  };
  activeEventSource?.close();
  activeEventSource = new EventSource(`/api/runs/${encodeURIComponent(id)}/events`);
  activeEventSource.onmessage = message => { try { consume(JSON.parse(message.data)); } catch { /* Polling recovers dropped events. */ } };
  try {
    while (alive) {
      const run = await request(`/api/runs/${encodeURIComponent(id)}`);
      (run.events || []).forEach(consume);
      if (run.status === "failed" || run.status === "interrupted") throw new Error(run.error || "The verification worker stopped before completion.");
      if (run.status === "completed") {
        if (run.stale || !run.loop_result) throw new Error("This run no longer matches the inputs. Verify the current version again.");
        await refreshProject(); candidateVisible = true; accepted = false; resultIdentity = ""; rebuildViewers(); renderResult();
        try { localStorage.removeItem(`motionfit.run.${project!.project_id}`); } catch { /* optional */ }
        return;
      }
      await new Promise(resolve => setTimeout(resolve, 1200));
    }
  } finally { activeEventSource?.close(); activeEventSource = null; }
}
function getTraces(): [TraceData | null, TraceData | null] {
  return [project?.loop_result?.initial_trace || null, project?.loop_result?.final_trace || null];
}
function visualDesign(design: DesignSpec): DesignSpec {
  return { ...design, visual_parts: Object.fromEntries((project?.catalog?.parts || []).map((p: any) => [p.part_id, p])) } as DesignSpec;
}
function rebuildViewers() {
  if (!project) return;
  const identity = JSON.stringify([project.design, project.loop_result?.final_design, project.loop_result?.final_metrics, candidateVisible, accepted, project.catalog]);
  if (identity === resultIdentity) return;
  resultIdentity = identity;
  beforeViewer?.dispose(); afterViewer?.dispose(); beforeViewer = null; afterViewer = null;
  $("viewport-before").replaceChildren(); $("viewport-after").replaceChildren();
  const result = project.loop_result;
  const [initialTrace, finalTrace] = getTraces();
  $("empty-view").hidden = true;
  $("after-card").hidden = !candidateVisible || !finalTrace;
  $("view-grid").classList.toggle("compare", candidateVisible && Boolean(finalTrace));
  $("before-label").textContent = accepted ? "Applied design · measured replay" : result ? "Initial design · measured replay" : "Initial geometry · not yet simulated";
  try {
    beforeViewer = new RobotArmViewport($("viewport-before"));
    const shownDesign = accepted ? result.final_design : result?.initial_design || project.design;
    const shownTrace = accepted ? finalTrace : initialTrace;
    beforeViewer.setDesignAndTrace(visualDesign(shownDesign), shownTrace || undefined, accepted ? result.passed : Boolean(result?.iterations?.[0]?.check?.passed), project.catalog);
    if (candidateVisible && finalTrace && result) {
      afterViewer = new RobotArmViewport($("viewport-after"));
      afterViewer.setDesignAndTrace(visualDesign(result.final_design), finalTrace, result.passed, project.catalog);
    }
    beforeViewer.setAxisOverlaysVisible($("axes").getAttribute("aria-pressed") === "true");
    beforeViewer.setGhostArmVisible($("ghost").getAttribute("aria-pressed") === "true");
    afterViewer?.setAxisOverlaysVisible($("axes").getAttribute("aria-pressed") === "true");
    afterViewer?.setGhostArmVisible($("ghost").getAttribute("aria-pressed") === "true");
  } catch (error) { $("empty-view").hidden = false; $("empty-view").textContent = `3D view unavailable: ${error instanceof Error ? error.message : error}`; }
  playhead = 0; playing = Boolean(initialTrace); lastTime = 0;
  $<HTMLInputElement>("timeline-slider").max = String(initialTrace?.t.at(-1) || 0);
  $<HTMLInputElement>("timeline-slider").step = "0.002";
  $("timeline-slider").toggleAttribute("disabled", !initialTrace); $("play").toggleAttribute("disabled", !initialTrace);
  $("play").textContent = playing ? "Ⅱ Pause" : "▶ Play";
  if (!shownHasTrace()) beforeViewer?.render();
}
function shownHasTrace() { return Boolean(project?.loop_result?.initial_trace?.t?.length); }
function renderResult() {
  const result = project?.loop_result;
  if (!result) { $("metrics").innerHTML = ""; refreshActions(); return; }
  const initial = result.initial_metrics; const final = result.final_metrics;
  $("metrics").innerHTML = [["Path error", `${num(initial.ee_rmse_mm)} → ${num(final.ee_rmse_mm)} mm`], ["Arm mass", `${num(initial.total_mass_kg, 3)} → ${num(final.total_mass_kg, 3)} kg`], ["Catalog cost", `$${num(initial.total_cost_usd)} → $${num(final.total_cost_usd)}`]].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
  const selected = result.iterations?.find((i: any) => i.design_id === result.final_design.design_id);
  const changes: string[] = [];
  for (const joint of ["J1", "J2", "J3", "J4"]) {
    const before = result.initial_design.joints[joint], after = result.final_design.joints[joint];
    if (JSON.stringify(before) !== JSON.stringify(after)) changes.push(`${joint}: ${before.actuator_id}${before.gearbox_id ? ` + ${before.gearbox_id}` : ""} → ${after.actuator_id}${after.gearbox_id ? ` + ${after.gearbox_id}` : ""}`);
  }
  if (JSON.stringify(result.initial_design.links) !== JSON.stringify(result.final_design.links)) changes.push(`Link geometry: ${JSON.stringify(result.final_design.links)}`);
  say("agent", `${result.passed ? "A candidate passes the configured numerical checks." : "No passing candidate was found within this bounded search."}\nSelected measured design: ${result.final_design.design_id}\nPath error: ${num(final.ee_rmse_mm)} mm. Arm mass: ${num(final.total_mass_kg, 3)} kg. Catalog cost: $${num(final.total_cost_usd)}.\n${changes.length ? `Measured candidate changes:\n${changes.join("\n")}` : "The selected design uses the original hardware."}${!result.passed ? `\nRemaining violations: ${(selected?.check?.violations || []).map((v: any) => v.check).join(", ")}` : "\nPreview the candidate, then apply it if you accept the result."}\nScope: arm-load and path tracking in the model; grasp success, collisions and mounting compatibility are not verified.`);
  setProgress(result.passed ? "Verification completed. Candidate ready for review." : "Verification completed without a passing candidate.");
  refreshActions();
}
function indexAt(trace: TraceData, t: number) {
  let lo = 0, hi = trace.t.length - 1;
  while (lo < hi) { const mid = Math.ceil((lo + hi) / 2); if (trace.t[mid] <= t) lo = mid; else hi = mid - 1; }
  return lo;
}
function telemetry(id: string, trace: TraceData, idx: number) {
  $(id).innerHTML = ["J1", "J2", "J3", "J4"].map((name, j) => {
    const demand = Math.abs(trace.tau_req[idx]?.[j] || 0), available = Math.abs(trace.tau_avail[idx]?.[j] || 0);
    const ratio = available > 1e-6 ? demand / available : demand > 1e-6 ? Infinity : 0;
    return `<span class="${ratio > 1 ? "overload" : ""}" title="Required torque / available torque at measured actual speed">${name} ${Number.isFinite(ratio) ? `${Math.round(ratio * 100)}%` : "over limit"}</span>`;
  }).join("");
}
function renderFrame() {
  const [a, b] = getTraces();
  if (a) {
    const first = accepted && b ? b : a;
    const idx = indexAt(first, playhead);
    beforeViewer?.updateFrame(first, idx); telemetry("live-strip-before", first, idx);
    if (b && afterViewer) { const j = indexAt(b, playhead); afterViewer.updateFrame(b, j); telemetry("live-strip-after", b, j); }
    $<HTMLInputElement>("timeline-slider").value = String(playhead);
    $("time-label").textContent = `${playhead.toFixed(2)} / ${(a.t.at(-1) || 0).toFixed(2)} s`;
  }
  beforeViewer?.render(); afterViewer?.render();
  const video = $<HTMLVideoElement>("source-video");
  if (a && video.src && video.readyState >= 2 && !busy && playing && Math.abs(video.currentTime - playhead) > 0.13 && !video.seeking) video.currentTime = Math.min(playhead, video.duration || playhead);
  if (!busy) drawSourceOverlay(a ? playhead : video.currentTime);
}
function drawSourceOverlay(t: number) {
  const canvas = $<HTMLCanvasElement>("skeleton-canvas");
  const ctx = canvas.getContext("2d"); if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!overlayFrames.length) return;
  let nearest = overlayFrames[0];
  for (const frame of overlayFrames) { if (Math.abs(frame.t - t) < Math.abs(nearest.t - t)) nearest = frame; }
  drawPoseOverlay(ctx, canvas.width, canvas.height, nearest.landmarks, $<HTMLSelectElement>("arm-side").value as "right" | "left");
}
function animate(now: number) {
  if (!alive) return;
  const duration = project?.loop_result?.initial_trace?.t?.at(-1) || 0;
  if (playing && !busy && lastTime) {
    playhead = Math.min(duration, playhead + Math.min(0.1, (now - lastTime) / 1000) * speed);
    if (playhead >= duration) { playing = false; $("play").textContent = "↺ Replay"; }
  }
  lastTime = now; renderFrame(); requestAnimationFrame(animate);
}
async function openCatalog() {
  await ensureProject();
  $<HTMLTextAreaElement>("catalog-json").value = JSON.stringify(project!.catalog, null, 2);
  $("catalog-summary").innerHTML = `<table><thead><tr><th>Part</th><th>Type</th><th>Mass</th><th>Cost</th><th>Source</th></tr></thead><tbody>${project!.catalog.parts.map(p => {
    const field = p.stall_torque_nm || p.ratio || p.mass_kg; const source = field?.src;
    const href = source?.doc ? `${apiPath("/datasheets/")}${encodeURIComponent(source.doc)}#page=${Number(source.page) || 1}` : "";
    return `<tr><td>${escape(p.part_id)}</td><td>${escape(p.type)}</td><td>${num(p.mass_kg?.value, 3)} kg</td><td>$${num(p.price_usd?.value)}</td><td>${href ? `<a href="${escape(href)}" target="_blank" rel="noopener">${escape(source.doc)} p.${Number(source.page) || 1}</a>` : "Missing source"}${p.continuous_torque_nm?.derived ? " · estimated continuous rating" : ""}</td></tr>`;
  }).join("")}</tbody></table>`;
  $<HTMLDialogElement>("catalog-dialog").showModal();
}

$("chat-form").addEventListener("submit", event => { event.preventDefault(); runAction(submitTask); });
$("prompt").addEventListener("keydown", event => { if ((event as KeyboardEvent).key === "Enter" && !(event as KeyboardEvent).shiftKey) { event.preventDefault(); $<HTMLFormElement>("chat-form").requestSubmit(); } });
$("chat-stream").addEventListener("scroll", () => { const el = $("chat-stream"); followChat = el.scrollHeight - el.scrollTop - el.clientHeight < 90; });
$("catalog-label").addEventListener("click", () => runAction(openCatalog));
$("confirm-catalog").addEventListener("click", () => runAction(async () => {
  const catalog = JSON.parse($<HTMLTextAreaElement>("catalog-json").value);
  if (!Array.isArray(catalog.parts) || !catalog.parts.length) throw new Error("Catalog must contain a nonempty parts array.");
  await request(apiPath("/catalog"), "PUT", catalog); await refreshProject();
  $<HTMLDialogElement>("catalog-dialog").close(); say("agent", "Saved your reviewed catalog. Previous verification and approval were invalidated. Click Verify motion to test these parts against the recorded target.");
  resultIdentity = ""; rebuildViewers(); $("metrics").innerHTML = "";
}));
$("catalog-file").addEventListener("change", async () => {
  const file = $<HTMLInputElement>("catalog-file").files?.[0]; if (!file) return;
  try { $<HTMLTextAreaElement>("catalog-json").value = JSON.stringify(JSON.parse(await file.text()), null, 2); } catch (e) { failure(e); }
});
$("upload-catalog").addEventListener("click", () => runAction(async () => {
  const files = $<HTMLInputElement>("datasheet-files").files; if (!files?.length) throw new Error("Choose PDFs and an optional prices.csv.");
  const data = new FormData(); for (const file of files) data.append("files", file);
  await request(apiPath("/catalog/files"), "POST", data); say("agent", "Source documents uploaded. Extract them, review the result and save the reviewed catalog before verification.");
}));
$("extract-pdfs").addEventListener("click", () => runAction(async () => {
  setProgress("Extracting specifications from your uploaded source documents…");
  const result = await request(apiPath("/catalog/extract?force_live=true"), "POST");
  $<HTMLTextAreaElement>("catalog-json").value = JSON.stringify(result.catalog, null, 2);
  await refreshProject(); say("agent", "Extraction completed. Review the actual extracted fields and sources in the catalog before saving.");
}));
$("input-video-file").addEventListener("change", () => {
  const file = $<HTMLInputElement>("input-video-file").files?.[0]; if (!file) return;
  if (file.size > 100 * 1024 * 1024) { failure(new Error("Use a video smaller than 100 MB.")); return; }
  sourceFile = file; overlayFrames = [];
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = URL.createObjectURL(file); $<HTMLVideoElement>("source-video").src = sourceUrl;
  $("video-wrap").hidden = false; $("source-empty").hidden = true;
  $("source-label").textContent = file.name;
  $("attachment").hidden = false; $("attachment").textContent = `Attached: ${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
});
$("import-landmarks").addEventListener("change", () => runAction(async () => {
  const file = $<HTMLInputElement>("import-landmarks").files?.[0]; if (!file) return;
  await ensureProject(); if (!project?.task) throw new Error("Describe your task and load before importing captured pose data.");
  const payload = JSON.parse(await file.text());
  await request(apiPath("/motion"), "POST", { landmarks: payload, orientation_constraint: $<HTMLSelectElement>("orientation").value || project?.task?.orientation_constraint || "free" });
  overlayFrames = payload.overlayFrames || []; await refreshProject();
  say("user", `Imported captured motion: ${file.name}`); say("agent", "Recorded pose data loaded and validated. Review the catalog, then verify this measured motion.");
  candidateVisible = false; accepted = false; resultIdentity = ""; rebuildViewers(); $("metrics").innerHTML = "";
}));
$("design-file").addEventListener("change", () => runAction(async () => {
  const file = $<HTMLInputElement>("design-file").files?.[0]; if (!file) return;
  await ensureProject(); await request(apiPath("/design"), "PUT", JSON.parse(await file.text())); await refreshProject();
  say("agent", `Imported initial arm from ${file.name}. Previous results and approval were invalidated.`); resultIdentity = ""; rebuildViewers();
}));
$("run").addEventListener("click", () => runAction(startRun));
$("compare").addEventListener("click", () => { candidateVisible = !candidateVisible; rebuildViewers(); });
$("accept").addEventListener("click", () => runAction(async () => {
  await request(apiPath("/accept"), "POST", { run_id: runId || undefined }); await refreshProject(); accepted = true; candidateVisible = false; rebuildViewers();
  say("agent", `Applied verified candidate ${project!.loop_result.final_design.design_id}. The current 3D arm now replays that candidate's measured joint angles. Review the report before sign-off.`);
}));
$("approve").addEventListener("click", () => $<HTMLDialogElement>("approval-dialog").showModal());
$("save-approval").addEventListener("click", () => runAction(async () => {
  const name = $<HTMLInputElement>("engineer").value.trim(); if (!name) throw new Error("Enter the reviewing engineer's name.");
  await request(apiPath("/approve"), "POST", { engineer_name: name, notes: $<HTMLTextAreaElement>("approval-notes").value }); await refreshProject();
  $<HTMLDialogElement>("approval-dialog").close(); say("agent", `Approval recorded for the current verified input version by ${name}. The MJCF and JSON bundle is available to download.`);
}));
$("report").addEventListener("click", () => { $<HTMLIFrameElement>("report-frame").src = apiPath("/report"); $<HTMLAnchorElement>("report-tab").href = apiPath("/report"); $<HTMLDialogElement>("report-dialog").showModal(); });
$("export").addEventListener("click", event => { if (!project?.approval?.approved) event.preventDefault(); });
document.querySelectorAll<HTMLElement>("[data-close]").forEach(button => button.addEventListener("click", () => $<HTMLDialogElement>(button.dataset.close!).close()));
$("new-project").addEventListener("click", () => runAction(async () => {
  beforeViewer?.dispose(); afterViewer?.dispose(); beforeViewer = null; afterViewer = null;
  project = null; messages = []; resultIdentity = ""; accepted = false; candidateVisible = false; playing = false; runId = ""; sourceFile = null; overlayFrames = [];
  if (sourceUrl) URL.revokeObjectURL(sourceUrl); sourceUrl = null;
  $<HTMLVideoElement>("source-video").removeAttribute("src"); $<HTMLVideoElement>("source-video").load(); $("video-wrap").hidden = true; $("source-empty").hidden = false; $("attachment").hidden = true;
  for (const id of ["payload", "budget", "mass", "reach", "prompt"]) ($<HTMLInputElement>(id)).value = "";
  await ensureProject(); renderMessages(); rebuildViewers(); $("metrics").innerHTML = ""; setProgress("New isolated project. Describe the task and attach your motion.");
}));
$("play").addEventListener("click", () => { if (playhead >= (getTraces()[0]?.t.at(-1) || 0)) playhead = 0; playing = !playing; $("play").textContent = playing ? "Ⅱ Pause" : "▶ Play"; });
$("timeline-slider").addEventListener("input", () => { playing = false; playhead = Number($<HTMLInputElement>("timeline-slider").value); $("play").textContent = "▶ Play"; const video = $<HTMLVideoElement>("source-video"); if (video.src && video.readyState >= 2) video.currentTime = Math.min(playhead, video.duration); renderFrame(); });
$("speed").addEventListener("change", () => speed = Number($<HTMLSelectElement>("speed").value));
for (const [id, setter] of [["axes", "setAxisOverlaysVisible"], ["ghost", "setGhostArmVisible"]] as const) $(id).addEventListener("click", () => {
  const value = $(id).getAttribute("aria-pressed") !== "true"; $(id).setAttribute("aria-pressed", String(value)); beforeViewer?.[setter](value); afterViewer?.[setter](value);
});
$<HTMLVideoElement>("source-video").addEventListener("loadedmetadata", () => { const video = $<HTMLVideoElement>("source-video"), canvas = $<HTMLCanvasElement>("skeleton-canvas"); canvas.width = video.videoWidth; canvas.height = video.videoHeight; });
$<HTMLVideoElement>("source-video").addEventListener("timeupdate", () => { if (!playing && !busy && getTraces()[0]) { playhead = Math.min($<HTMLVideoElement>("source-video").currentTime, getTraces()[0]!.t.at(-1)!); renderFrame(); } });
window.addEventListener("beforeunload", () => { alive = false; activeAbort?.abort(); activeEventSource?.close(); beforeViewer?.dispose(); afterViewer?.dispose(); if (sourceUrl) URL.revokeObjectURL(sourceUrl); });

async function boot() {
  renderMessages(); setProgress(progress);
  let stored: string | null = null; try { stored = localStorage.getItem("motionfit.project"); } catch { /* optional */ }
  if (stored && /^proj-[a-zA-Z0-9-]+$/.test(stored)) {
    try {
      project = await request<Project>(`/api/projects/${encodeURIComponent(stored)}`);
      try { const saved = JSON.parse(localStorage.getItem(historyKey()) || "[]"); messages = Array.isArray(saved) ? saved.filter(m => ["user", "agent"].includes(m.role) && typeof m.text === "string") : []; } catch { messages = []; }
      accepted = Boolean(project.accepted || project.workflow?.accepted);
      overlayFrames = project.landmarks?.overlayFrames || [];
      const c = project.task?.constraints || {};
      for (const [id, key] of [["payload", "payload_kg"], ["budget", "max_cost_usd"], ["mass", "max_total_mass_kg"], ["reach", "min_reach_m"]]) if (c[key] != null) $<HTMLInputElement>(id).value = String(c[key]);
      if (project.task?.orientation_constraint) $<HTMLSelectElement>("orientation").value = project.task.orientation_constraint;
      refreshActions(); rebuildViewers(); renderMessages();
      if (project.landmarks) $("source-empty").textContent = "Captured pose data is saved. Reattach the original video for synchronized source playback; browsers do not persist local file access.";
      let pending: string | null = null; try { pending = localStorage.getItem(`motionfit.run.${project.project_id}`); } catch { /* optional */ }
      if (pending) { runId = pending; setBusy(true); try { await watchRun(pending); } finally { setBusy(false); } }
      else if (project.loop_result) { const r = project.loop_result; $("metrics").innerHTML = `<div><span>Measured path error</span><strong>${num(r.final_metrics.ee_rmse_mm)} mm</strong></div><div><span>Arm mass</span><strong>${num(r.final_metrics.total_mass_kg, 3)} kg</strong></div><div><span>Catalog cost</span><strong>$${num(r.final_metrics.total_cost_usd)}</strong></div>`; }
    } catch (error) { failure(error); }
  }
}
requestAnimationFrame(animate);
boot().catch(failure);
