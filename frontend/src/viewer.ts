/** Parametric part envelopes and exact replay of the backend simulation trace. */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export interface TraceData {
  fps: number; t: number[];
  q_des: number[][]; qd_des: number[][]; q_act: number[][]; qd_act: number[][];
  tau_req: number[][]; tau_cmd: number[][]; tau_avail: number[][];
  ee_des: number[][]; ee_act: number[][];
}
export interface DesignSpec {
  design_id: string; topology: string;
  links: { base_height_m: number; upper_arm_m: number; forearm_m: number; wrist_to_tool_m: number; tube_profile: string };
  joints: Record<string, { actuator_id: string; gearbox_id: string | null }>;
  payload_kg: number;
  constraints: { max_total_mass_kg: number; max_cost_usd: number; min_reach_m: number; link_length_bounds_m: Record<string, [number, number]> };
  thresholds: { ee_rmse_mm: number; peak_torque_ratio: number; rms_torque_ratio: number; peak_speed_ratio: number };
}
export interface CatalogPartVisual {
  part_id: string; dimensions_mm?: number[] | { value: number[] }; series?: string; type?: string;
}
export interface VisualCatalog { parts?: CatalogPartVisual[]; items?: CatalogPartVisual[] }
const JOINTS = ["J1", "J2", "J3", "J4"] as const;

/** Catalog bounding envelope in metres. Missing dimensions remain unknown. */
export function partEnvelope(part?: CatalogPartVisual): [number, number, number] | null {
  const dimensions = part?.dimensions_mm;
  const values = Array.isArray(dimensions) ? dimensions : dimensions?.value;
  if (!values || values.length !== 3 || !values.every(v => Number.isFinite(v) && v > 0)) return null;
  return [values[0] / 1000, values[1] / 1000, values[2] / 1000];
}
export function tubeRadius(profile: string): number {
  const match = /tube_(\d+(?:\.\d+)?)x/.exec(profile);
  if (!match) throw new Error(`Unsupported tube dimensions: ${profile}`);
  return Number(match[1]) / 2000;
}
/** No frontend IK or success flag participates in frame selection. */
export function resolveTraceFrame(trace: TraceData, frameIdx: number) {
  if (!trace.t.length) throw new Error("The simulation trace has no frames.");
  const index = Math.max(0, Math.min(trace.t.length - 1, Math.trunc(frameIdx)));
  if (!Number.isFinite(index)) throw new Error("The simulation frame index is invalid.");
  const read = (field: "q_act" | "q_des" | "ee_act" | "ee_des", width: number) => {
    const values = trace[field]?.[index];
    if (!values || values.length !== width || !values.every(Number.isFinite)) throw new Error(`Invalid simulation trace: ${field}[${index}].`);
    return values;
  };
  const qAct = read("q_act", 4), qDes = read("q_des", 4), eeAct = read("ee_act", 3), eeDes = read("ee_des", 3);
  const effortRatios = JOINTS.map((_, j) => {
    const requested = trace.tau_req[index]?.[j], available = trace.tau_avail[index]?.[j];
    if (!Number.isFinite(requested) || !Number.isFinite(available)) return null;
    if (available === 0) return requested === 0 ? 0 : Infinity;
    return Math.abs(requested / available);
  });
  return { index, time: trace.t[index], qAct, qDes, eeAct, eeDes, effortRatios };
}
interface JointVisual { pivot: THREE.Group; overlay: THREE.Group; effortMaterial: THREE.MeshStandardMaterial }
interface RobotRig { root: THREE.Group; base: THREE.Group; links: THREE.Group[]; joints: Record<typeof JOINTS[number], JointVisual> }
function standardMaterial(color: number, ghost: boolean): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color: ghost ? 0x38bdf8 : color, roughness: 0.4, metalness: ghost ? 0 : 0.6, transparent: ghost, opacity: ghost ? 0.16 : 1, depthWrite: !ghost });
}
function tube(length: number, radius: number, material: THREE.Material, ghost: boolean): THREE.Group {
  const group = new THREE.Group();
  const geometry = new THREE.CylinderGeometry(radius, radius, length, 24);
  geometry.rotateZ(-Math.PI / 2); geometry.translate(length / 2, 0, 0);
  const mesh = new THREE.Mesh(geometry, material); mesh.castShadow = !ghost;
  group.add(mesh); return group;
}
function jointOverlay(radius: number, material: THREE.MeshStandardMaterial, zAxis: boolean): THREE.Group {
  const group = new THREE.Group();
  const ring = new THREE.Mesh(new THREE.TorusGeometry(radius, radius * 0.07, 6, 40, Math.PI * 1.6), material);
  if (!zAxis) ring.rotation.x = Math.PI / 2;
  group.add(ring); return group;
}
function buildRig(design: DesignSpec, catalog: VisualCatalog, ghost: boolean): RobotRig {
  const root = new THREE.Group(), base = new THREE.Group(); root.add(base);
  const { base_height_m: baseHeight, upper_arm_m: upper, forearm_m: forearm, wrist_to_tool_m: tool } = design.links;
  if (![baseHeight, upper, forearm, tool].every(v => Number.isFinite(v) && v > 0)) throw new Error("Invalid robot link dimensions.");
  for (const key of JOINTS) if (!design.joints[key]?.actuator_id) throw new Error(`Robot design is missing ${key}.`);
  const radius = tubeRadius(design.links.tube_profile);
  const structure = standardMaterial(design.links.tube_profile.startsWith("cf_") ? 0x334155 : 0xcbd5e1, ghost);
  const housing = standardMaterial(0x64748b, ghost), gearMaterial = standardMaterial(0xd5a54e, ghost);
  const unknownMaterial = new THREE.MeshStandardMaterial({ color: 0x94a3b8, wireframe: true, transparent: true, opacity: ghost ? 0.1 : 0.45 });
  root.userData.ownedMaterials = [structure, housing, gearMaterial, unknownMaterial];
  const baseGeometry = new THREE.CylinderGeometry(radius * 1.4, radius * 1.4, baseHeight, 24); baseGeometry.rotateX(Math.PI / 2);
  const baseMesh = new THREE.Mesh(baseGeometry, structure); baseMesh.position.z = baseHeight / 2; base.add(baseMesh);
  const lengths = [upper, forearm, tool], links: THREE.Group[] = [], joints = {} as RobotRig["joints"];
  const parts = catalog.parts ?? catalog.items ?? [];
  let parent: THREE.Group = root;
  JOINTS.forEach((key, i) => {
    const pivot = new THREE.Group();
    if (i === 1) pivot.position.z = baseHeight;
    if (i === 2) pivot.position.x = upper;
    if (i === 3) pivot.position.x = forearm;
    parent.add(pivot);
    const spec = design.joints[key];
    if (!spec) throw new Error(`Robot design is missing ${key}.`);
    const envelope = partEnvelope(parts.find(p => p.part_id === spec.actuator_id));
    // Unknown dimensions use a wire marker, making no part-size claim.
    const body = new THREE.Mesh(envelope ? new THREE.BoxGeometry(...envelope) : new THREE.IcosahedronGeometry(0.012, 0), envelope ? housing : unknownMaterial);
    body.userData = { part_id: spec.actuator_id, representation: envelope ? "catalog_bounding_envelope" : "unknown_dimensions_marker" };
    if (i === 0) body.position.z = baseHeight / 2;
    body.castShadow = !ghost; pivot.add(body);
    if (spec.gearbox_id) {
      const gearEnvelope = partEnvelope(parts.find(p => p.part_id === spec.gearbox_id));
      const gearMesh = new THREE.Mesh(gearEnvelope ? new THREE.BoxGeometry(...gearEnvelope) : new THREE.IcosahedronGeometry(0.01, 0), gearEnvelope ? gearMaterial : unknownMaterial);
      gearMesh.position.y = (envelope?.[1] ?? 0) / 2 + (gearEnvelope?.[1] ?? 0) / 2;
      if (i === 0) gearMesh.position.z = baseHeight / 2;
      gearMesh.userData = { part_id: spec.gearbox_id, representation: gearEnvelope ? "catalog_bounding_envelope" : "unknown_dimensions_marker", mounting_verified: false };
      pivot.add(gearMesh);
    }
    const effortMaterial = new THREE.MeshStandardMaterial({ color: 0x22c55e, emissive: 0x16a34a, emissiveIntensity: 0.4 });
    root.userData.ownedMaterials.push(effortMaterial);
    const overlay = ghost ? new THREE.Group() : jointOverlay(Math.max(...(envelope ?? [0.025])) * 0.7, effortMaterial, i === 0);
    if (i === 0) overlay.position.z = baseHeight / 2;
    pivot.add(overlay); joints[key] = { pivot, overlay, effortMaterial };
    if (i > 0) { const link = tube(lengths[i - 1], i === 3 ? 0.012 : radius, structure, ghost); pivot.add(link); links.push(link); }
    parent = pivot;
  });
  return { root, base, links, joints };
}
/** Dispose each GPU resource owned by a removed scene subtree. */
function releaseTree(root: THREE.Object3D): void {
  const geometries = new Set<THREE.BufferGeometry>(), materials = new Set<THREE.Material>(), textures = new Set<THREE.Texture>();
  root.traverse(object => {
    const mesh = object as THREE.Mesh;
    if (mesh.geometry) geometries.add(mesh.geometry);
    const list = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
    list.push(...(object.userData.ownedMaterials ?? []));
    list.forEach(material => { materials.add(material); for (const value of Object.values(material)) if (value instanceof THREE.Texture) textures.add(value); });
    if (object instanceof THREE.DirectionalLight || object instanceof THREE.SpotLight) object.shadow.dispose();
  });
  textures.forEach(texture => texture.dispose()); materials.forEach(material => material.dispose()); geometries.forEach(geometry => geometry.dispose());
  root.removeFromParent();
}
export class RobotArmViewport {
  private container: HTMLElement;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  public controls: OrbitControls;
  private worldZUp = new THREE.Group();
  private actualRig: RobotRig | null = null;
  private ghostRig: RobotRig | null = null;
  private targetLine: THREE.Line | null = null;
  private actualLine: THREE.Line | null = null;
  private errorTetherLine: THREE.Line | null = null;
  private eeActualMarker: THREE.Mesh;
  private eeTargetMarker: THREE.Mesh;
  private showAxisOverlays = true;
  private showGhostArm = true;
  private resizeObserver: ResizeObserver;
  private disposed = false;
  private hasTrace = false;
  private lastWidth = 0;
  private lastHeight = 0;
  constructor(container: HTMLElement) {
    this.container = container; this.scene.background = new THREE.Color(0xe9ecec);
    const width = Math.max(320, container.clientWidth), height = Math.max(240, container.clientHeight);
    this.camera = new THREE.PerspectiveCamera(40, width / height, 0.005, 30); this.camera.position.set(0.9, 0.65, 0.9);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); this.renderer.setSize(width, height, false);
    this.renderer.shadowMap.enabled = true; this.renderer.shadowMap.type = THREE.PCFSoftShadowMap; this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    container.replaceChildren(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement); this.controls.enableDamping = true; this.controls.dampingFactor = 0.08;
    this.controls.target.set(0.25, 0.2, 0); this.controls.update();
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.4));
    const key = new THREE.DirectionalLight(0xffffff, 2); key.position.set(1.4, 2.4, 1.6); key.castShadow = true; this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xcbd5e1, 0.8); fill.position.set(-1.5, 1.2, -1.2); this.scene.add(fill);
    const floorGeometry = new THREE.PlaneGeometry(4, 4); floorGeometry.rotateX(-Math.PI / 2);
    const floor = new THREE.Mesh(floorGeometry, new THREE.MeshStandardMaterial({ color: 0xf4f6f7, roughness: 0.9 }));
    floor.position.y = -0.005; floor.receiveShadow = true; this.scene.add(floor, new THREE.GridHelper(4, 40, 0xb8bec2, 0xd8dddf));
    this.worldZUp.rotation.x = -Math.PI / 2; this.scene.add(this.worldZUp);
    // EE point measurements do not claim an unmodelled physical grasp.
    this.eeActualMarker = new THREE.Mesh(new THREE.SphereGeometry(0.009, 16, 12), new THREE.MeshBasicMaterial({ color: 0x22c55e }));
    this.eeTargetMarker = new THREE.Mesh(new THREE.SphereGeometry(0.011, 16, 12), new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true }));
    this.eeActualMarker.visible = false; this.eeTargetMarker.visible = false; this.worldZUp.add(this.eeActualMarker, this.eeTargetMarker);
    this.resizeObserver = new ResizeObserver(() => this.handleResize()); this.resizeObserver.observe(container);
  }
  public handleResize(): void {
    if (this.disposed) return;
    const width = Math.floor(this.container.clientWidth), height = Math.floor(this.container.clientHeight);
    if (width <= 10 || height <= 10 || (width === this.lastWidth && height === this.lastHeight)) return;
    this.lastWidth = width; this.lastHeight = height; this.camera.aspect = width / height; this.camera.updateProjectionMatrix(); this.renderer.setSize(width, height, false);
  }
  public setDesignAndTrace(design: DesignSpec, trace: TraceData | null = null, passed = false, catalog: VisualCatalog = {}): void {
    if (this.disposed) return;
    if (trace) {
      resolveTraceFrame(trace, 0);
      if (trace.t.some((time, i) => !Number.isFinite(time) || (i > 0 && time <= trace.t[i - 1]))) throw new Error("Invalid simulation trace timestamps.");
      for (const field of ["q_des", "q_act", "ee_des", "ee_act"] as const) {
        const width = field.startsWith("q_") ? 4 : 3;
        if (trace[field].length !== trace.t.length || trace[field].some(point => point.length !== width || !point.every(Number.isFinite))) throw new Error(`Invalid simulation trace: ${field} does not match timestamps.`);
      }
    }
    const actualRig = buildRig(design, catalog, false), ghostRig = buildRig(design, catalog, true);
    for (const object of [this.actualRig?.root, this.ghostRig?.root, this.targetLine, this.actualLine, this.errorTetherLine]) if (object) releaseTree(object);
    this.hasTrace = !!trace;
    this.targetLine = this.actualLine = this.errorTetherLine = null;
    this.actualRig = actualRig; this.ghostRig = ghostRig; this.ghostRig.root.visible = this.showGhostArm;
    this.worldZUp.add(this.actualRig.root, this.ghostRig.root);
    this.setAxisOverlaysVisible(this.showAxisOverlays);
    if (!trace) {
      this.ghostRig.root.visible = false;
      this.eeActualMarker.visible = false;
      this.eeTargetMarker.visible = false;
      this.fitCamera(design, null);
      return;
    }
    const path = (points: number[][], color: number) => new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map(point => new THREE.Vector3(...point))), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.75 }));
    this.targetLine = path(trace.ee_des, 0x38bdf8); this.targetLine.visible = this.showGhostArm; this.actualLine = path(trace.ee_act, passed ? 0x10b981 : 0xef4444);
    this.errorTetherLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.8 }));
    this.worldZUp.add(this.targetLine, this.actualLine, this.errorTetherLine);
    this.eeActualMarker.visible = true; this.eeTargetMarker.visible = this.showGhostArm; this.setAxisOverlaysVisible(this.showAxisOverlays); this.updateFrame(trace, 0); this.fitCamera(design, trace);
  }
  private fitCamera(design: DesignSpec, trace: TraceData | null): void {
    const box = new THREE.Box3(new THREE.Vector3(-0.05, -0.05, 0), new THREE.Vector3(0.05, 0.05, design.links.base_height_m));
    for (const point of [...(trace?.ee_des ?? []), ...(trace?.ee_act ?? [])]) box.expandByPoint(new THREE.Vector3(...point));
    if (!trace) box.expandByPoint(new THREE.Vector3(design.links.upper_arm_m + design.links.forearm_m + design.links.wrist_to_tool_m, 0, design.links.base_height_m));
    const center = box.getCenter(new THREE.Vector3()), reach = design.links.upper_arm_m + design.links.forearm_m + design.links.wrist_to_tool_m;
    const distance = Math.max(box.getSize(new THREE.Vector3()).length(), reach, 0.3) * 1.8;
    const target = new THREE.Vector3(center.x, center.z, -center.y); this.controls.target.copy(target);
    this.camera.position.copy(target).add(new THREE.Vector3(0.8, 0.65, 1).normalize().multiplyScalar(distance));
    this.camera.far = Math.max(30, distance * 5); this.camera.updateProjectionMatrix(); this.controls.update();
  }
  /** Assembly inspection does not imply physical assembly validation. */
  public setAssemblyStage(stage: number): void {
    if (!this.actualRig) return;
    this.actualRig.base.visible = stage >= 1; this.actualRig.joints.J1.pivot.visible = stage >= 2; this.actualRig.joints.J2.pivot.visible = stage >= 3;
    this.actualRig.joints.J3.pivot.visible = stage >= 5; this.actualRig.joints.J4.pivot.visible = stage >= 6;
    this.actualRig.links.forEach((link, i) => { link.visible = stage >= 3 + i; });
  }
  public setAxisOverlaysVisible(visible: boolean): void {
    this.showAxisOverlays = visible; if (this.actualRig) JOINTS.forEach(key => { this.actualRig!.joints[key].overlay.visible = visible; });
  }
  public setGhostArmVisible(visible: boolean): void {
    this.showGhostArm = visible; if (this.ghostRig) this.ghostRig.root.visible = visible && this.hasTrace;
    this.eeTargetMarker.visible = visible && this.hasTrace; if (this.targetLine) this.targetLine.visible = visible; if (this.errorTetherLine) this.errorTetherLine.visible = visible;
  }
  /** A label does not create a fabricated physical object or grasp animation. */
  public setTargetObject(objectName: string): void { this.eeActualMarker.userData.task_object_label = objectName; }
  public updateFrame(trace: TraceData, frameIdx: number): void {
    if (!this.actualRig || this.disposed) return;
    const frame = resolveTraceFrame(trace, frameIdx);
    const pose = (rig: RobotRig, angles: number[]) => {
      rig.joints.J1.pivot.rotation.z = angles[0]; rig.joints.J2.pivot.rotation.y = -angles[1]; rig.joints.J3.pivot.rotation.y = -angles[2]; rig.joints.J4.pivot.rotation.y = -angles[3];
    };
    pose(this.actualRig, frame.qAct); if (this.ghostRig) pose(this.ghostRig, frame.qDes);
    this.eeActualMarker.position.fromArray(frame.eeAct); this.eeTargetMarker.position.fromArray(frame.eeDes);
    if (this.errorTetherLine) {
      const positions = this.errorTetherLine.geometry.getAttribute("position") as THREE.BufferAttribute;
      positions.setXYZ(0, ...frame.eeDes as [number, number, number]); positions.setXYZ(1, ...frame.eeAct as [number, number, number]);
      positions.needsUpdate = true; this.errorTetherLine.geometry.computeBoundingSphere();
    }
    JOINTS.forEach((key, i) => {
      const ratio = frame.effortRatios[i], color = ratio === null ? 0x94a3b8 : ratio > 1 ? 0xef4444 : ratio >= 0.8 ? 0xf59e0b : 0x22c55e;
      const material = this.actualRig!.joints[key].effortMaterial; material.color.setHex(color); material.emissive.setHex(color);
    });
  }
  public render(): void { if (this.disposed) return; this.controls.update(); this.renderer.render(this.scene, this.camera); }
  public syncCameraFrom(other: RobotArmViewport): void {
    if (this.disposed) return; this.camera.position.copy(other.camera.position); this.camera.quaternion.copy(other.camera.quaternion); this.controls.target.copy(other.controls.target); this.controls.update();
  }
  public dispose(): void {
    if (this.disposed) return;
    this.disposed = true; this.resizeObserver.disconnect(); this.controls.dispose(); releaseTree(this.scene); this.scene.clear();
    this.renderer.renderLists.dispose(); this.renderer.dispose(); this.renderer.forceContextLoss(); this.renderer.domElement.remove();
    this.actualRig = null; this.ghostRig = null; this.targetLine = this.actualLine = this.errorTetherLine = null;
  }
}
