/** Actual MediaPipe pose measurements. Tracking failures are reported, never replaced by a demo. */
export type Point3 = [number, number, number];
export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
  visibility: number;
}
export interface LandmarkFrame {
  t: number;
  visibility: number;
  shoulder: Point3;
  elbow: Point3;
  wrist: Point3;
  index: Point3;
  hip: Point3;
  confidence: Record<"shoulder" | "elbow" | "wrist" | "index" | "hip", number>;
}
export interface PoseOverlayFrame {
  t: number;
  landmarks: LandmarkPoint[];
}
export interface LandmarksPayload {
  fps: number;
  duration_s: number;
  arm_side: "right" | "left";
  source: "mediapipe_pose_world";
  coordinate_system: "mediapipe_world_m";
  frames: LandmarkFrame[];
  overlayFrames: PoseOverlayFrame[];
  tracking: { sampled_frames: number; detected_frames: number; missing_fraction: number };
}
export interface PoseExtractionOptions {
  videoEl?: HTMLVideoElement;
  overlayCanvas?: HTMLCanvasElement;
  armSide?: "right" | "left";
  fps?: number;
  maxDurationS?: number;
  onProgress?: (pct: number, statusText: string) => void;
  signal?: AbortSignal;
}

/** Select raw measured coordinates. Missing or occluded points produce no frame. */
export function measuredPoseFrame(
  world: Array<{ x: number; y: number; z: number; visibility?: number }> | undefined,
  normalized: Array<{ visibility?: number }> | undefined,
  t: number,
  armSide: "right" | "left" = "right",
): LandmarkFrame | null {
  const names = ["shoulder", "elbow", "wrist", "index", "hip"] as const;
  const indices = armSide === "right" ? [12, 14, 16, 20, 24] : [11, 13, 15, 19, 23];
  const confidence = Object.fromEntries(names.map((name, k) => [name,
    Math.min(world?.[indices[k]]?.visibility ?? 0, normalized?.[indices[k]]?.visibility ?? 0),
  ])) as LandmarkFrame["confidence"];
  const visibility = Math.min(...Object.values(confidence));
  if (visibility < 0.5 || !world || !indices.every(i => [world[i]?.x, world[i]?.y, world[i]?.z].every(Number.isFinite))) return null;
  const coordinates = Object.fromEntries(names.map((name, k) => {
    const p = world[indices[k]];
    return [name, [p.x, p.y, p.z] as Point3];
  })) as Pick<LandmarkFrame, typeof names[number]>;
  return { t, visibility, confidence, ...coordinates };
}

export function validatePoseTracking(sampledFrames: number, detectedFrames: number, armSide: "right" | "left" = "right"): number {
  const missing = 1 - detectedFrames / sampledFrames;
  if (!Number.isFinite(missing) || missing > 0.2 || detectedFrames < 15) throw new Error(`Pose tracking failed: ${(100 * missing).toFixed(1)}% of frames lacked a visible ${armSide} arm and hip. Keep the full arm and torso in view, improve lighting, or select the other arm. No substitute motion was generated.`);
  return missing;
}

function abortIfNeeded(signal?: AbortSignal): void {
  if (signal?.aborted) throw new DOMException("Video analysis cancelled.", "AbortError");
}

/** Register listeners before the operation, with a bounded wait and abort cleanup. */
function waitForVideo(
  video: HTMLVideoElement,
  eventName: "loadedmetadata" | "loadeddata" | "seeked",
  action: () => void,
  signal?: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener(eventName, ready);
      video.removeEventListener("error", failed);
      signal?.removeEventListener("abort", cancelled);
    };
    const ready = () => { cleanup(); resolve(); };
    const failed = () => { cleanup(); reject(new Error("The browser could not decode this video. Use an MP4 with H.264 video.")); };
    const cancelled = () => { cleanup(); reject(new DOMException("Video analysis cancelled.", "AbortError")); };
    const timer = window.setTimeout(() => { cleanup(); reject(new Error(`Timed out while reading the video (${eventName}).`)); }, 15000);
    video.addEventListener(eventName, ready, { once: true });
    video.addEventListener("error", failed, { once: true });
    signal?.addEventListener("abort", cancelled, { once: true });
    if (signal?.aborted) { cancelled(); return; }
    try { action(); } catch (error) { cleanup(); reject(error); }
  });
}

export function drawPoseOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  landmarks: LandmarkPoint[],
  armSide: "right" | "left" = "right",
): void {
  const [shoulder, elbow, wrist, index, hip] = armSide === "right" ? [12, 14, 16, 20, 24] : [11, 13, 15, 19, 23];
  const visible = (i: number) => landmarks[i] && landmarks[i].visibility >= 0.5;
  ctx.save();
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = Math.max(2, width / 240);
  for (const [a, b] of [[hip, shoulder], [shoulder, elbow], [elbow, wrist], [wrist, index]]) {
    if (!visible(a) || !visible(b)) continue;
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * width, landmarks[a].y * height);
    ctx.lineTo(landmarks[b].x * width, landmarks[b].y * height);
    ctx.stroke();
  }
  ctx.fillStyle = "#22c55e";
  for (const i of [shoulder, elbow, wrist, index, hip]) {
    if (!visible(i)) continue;
    ctx.beginPath();
    ctx.arc(landmarks[i].x * width, landmarks[i].y * height, Math.max(3, width / 150), 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.restore();
}

/**
 * Return unscaled MediaPipe world coordinates in metres. The backend owns the
 * camera-to-robot transform and anthropometric scaling. Low-confidence samples
 * are omitted with their actual timestamps and an explicit missing-frame count.
 * A supplied video element's existing preview source is restored on completion.
 */
export async function extractLandmarksFromVideoFile(
  file: File,
  options: PoseExtractionOptions = {},
): Promise<LandmarksPayload> {
  const { overlayCanvas, onProgress, signal } = options;
  const armSide = options.armSide ?? "right";
  const fps = options.fps ?? 30;
  if (!Number.isFinite(fps) || fps < 5 || fps > 60) throw new Error("Pose sampling rate must be between 5 and 60 fps.");
  const video = options.videoEl ?? document.createElement("video");
  const previousSrc = video.getAttribute("src");
  const previousTime = video.currentTime;
  const previousMuted = video.muted;
  const wasPaused = video.paused;
  const objectUrl = URL.createObjectURL(file);
  let detector: import("@mediapipe/tasks-vision").PoseLandmarker | undefined;
  video.pause();
  video.muted = true;
  video.playsInline = true;
  try {
    abortIfNeeded(signal);
    onProgress?.(0, "Reading video metadata…");
    await waitForVideo(video, "loadedmetadata", () => { video.src = objectUrl; video.load(); }, signal);
    const duration = video.duration;
    const maxDuration = options.maxDurationS ?? 60;
    if (!Number.isFinite(duration) || duration < 0.5) throw new Error("Video must contain at least 0.5 seconds of decodable motion.");
    if (duration > maxDuration) throw new Error(`Video is ${duration.toFixed(1)} seconds long. Trim it to ${maxDuration} seconds or less; no frames were silently discarded.`);
    if (video.readyState < 2) await waitForVideo(video, "loadeddata", () => {}, signal);
    if (overlayCanvas) {
      overlayCanvas.width = video.videoWidth;
      overlayCanvas.height = video.videoHeight;
    }
    const ctx = overlayCanvas?.getContext("2d");
    onProgress?.(2, "Loading MediaPipe pose model…");
    const vision = await import("@mediapipe/tasks-vision");
    const fileset = await vision.FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm");
    abortIfNeeded(signal);
    detector = await vision.PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    const frames: LandmarkFrame[] = [];
    const overlayFrames: PoseOverlayFrame[] = [];
    const total = Math.max(15, Math.floor(duration * fps));
    for (let i = 0; i < total; i++) {
      abortIfNeeded(signal);
      const t = Math.min(i / fps, duration - 0.001);
      // Seeking to an already decoded t=0 does not fire seeked in several browsers.
      if (Math.abs(video.currentTime - t) > 0.0001 || video.readyState < 2) {
        await waitForVideo(video, "seeked", () => { video.currentTime = t; }, signal);
      }
      if (ctx && overlayCanvas) ctx.drawImage(video, 0, 0, overlayCanvas.width, overlayCanvas.height);
      const result = detector.detectForVideo(video, i * 1000 / fps);
      const world = result.worldLandmarks[0];
      const normalized = result.landmarks[0];
      if (normalized) {
        const points = normalized.map(p => ({ x: p.x, y: p.y, z: p.z, visibility: p.visibility ?? 0 }));
        overlayFrames.push({ t, landmarks: points });
        if (ctx && overlayCanvas) drawPoseOverlay(ctx, overlayCanvas.width, overlayCanvas.height, points, armSide);
      } else overlayFrames.push({ t, landmarks: [] });
      const measurement = measuredPoseFrame(world, normalized, t, armSide);
      if (measurement) frames.push(measurement);
      if (i % 5 === 0 || i === total - 1) {
        onProgress?.(Math.round(5 + 95 * (i + 1) / total), `Measured ${frames.length}/${i + 1} frames (${armSide} arm)…`);
        await new Promise(resolve => window.setTimeout(resolve, 0));
      }
    }
    const missing = validatePoseTracking(total, frames.length, armSide);
    return {
      fps, duration_s: duration, arm_side: armSide, source: "mediapipe_pose_world", coordinate_system: "mediapipe_world_m",
      frames, overlayFrames,
      tracking: { sampled_frames: total, detected_frames: frames.length, missing_fraction: missing },
    };
  } catch (error) {
    if (error instanceof Error) throw error;
    throw new Error("Video pose extraction failed. No substitute motion was generated.");
  } finally {
    detector?.close();
    video.pause();
    if (previousSrc) {
      video.src = previousSrc;
      if (Number.isFinite(previousTime)) {
        const restoreTime = () => { video.currentTime = previousTime; };
        video.addEventListener("loadedmetadata", restoreTime, { once: true });
      }
      video.load();
      if (!wasPaused) void video.play().catch(() => {});
    } else {
      video.removeAttribute("src");
      video.load();
    }
    video.muted = previousMuted;
    URL.revokeObjectURL(objectUrl);
  }
}
