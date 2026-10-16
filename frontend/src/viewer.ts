/**
 * CAD/CAE 3D Robot Arm & Floor-to-Floor Water-Bottle Transfer Simulation Viewer (Three.js)
 *
 * Models a realistic 4-DOF desktop robot arm with:
 * - Sculpted metallic/anodized motor housings & industrial base controller
 * - Translucent blue joint-axis cylinders & color-coded strain/effort arrows
 * - 2-finger parallel bottle gripper
 * - Floor Pickup Spot A (z = 0) and Far Floor Target Spot B (z = 0) separated by a long 0.68 m span
 * - 3D 500 mL Water Bottle sitting directly on the floor at Spot A, lifted high across a wide arc,
 *   and placed flat on the floor at Spot B (or sagging down to the floor mid-way when the un-upgraded
 *   shoulder motor runs out of strength).
 */

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export interface TraceData {
  fps: number;
  t: number[];
  q_des: number[][];
  qd_des: number[][];
  q_act: number[][];
  qd_act: number[][];
  tau_req: number[][];
  tau_cmd: number[][];
  tau_avail: number[][];
  ee_des: number[][];
  ee_act: number[][];
}

export interface DesignSpec {
  design_id: string;
  topology: string;
  links: {
    base_height_m: number;
    upper_arm_m: number;
    forearm_m: number;
    wrist_to_tool_m: number;
    tube_profile: string;
  };
  joints: Record<string, { actuator_id: string; gearbox_id: string | null }>;
  payload_kg: number;
  constraints: {
    max_total_mass_kg: number;
    max_cost_usd: number;
    min_reach_m: number;
    link_length_bounds_m: Record<string, [number, number]>;
  };
  thresholds: {
    ee_rmse_mm: number;
    peak_torque_ratio: number;
    rms_torque_ratio: number;
    peak_speed_ratio: number;
  };
}

interface JointVisualHandles {
  pivot: THREE.Group;
  housingMesh: THREE.Mesh;
  axisOverlayGroup: THREE.Group;
  arrowMeshes: THREE.Mesh[];
  arrowMat: THREE.MeshStandardMaterial;
  housingAccentMat: THREE.MeshStandardMaterial;
}

interface RobotRig {
  root: THREE.Group;
  baseGroup: THREE.Group;
  upperArmGroup: THREE.Group;
  forearmGroup: THREE.Group;
  wristGripperGroup: THREE.Group;
  leftFinger: THREE.Mesh | null;
  rightFinger: THREE.Mesh | null;
  joints: Record<"J1" | "J2" | "J3" | "J4", JointVisualHandles>;
}

/**
 * Floor-to-Floor Water Bottle Coordinates in Z-up world space (meters):
 * - The bottle's bottom sits flat on the floor mat (z = 0.004 m) when its neck grasp point is at z = 0.088 m.
 * - Spot A (Floor Pickup) is far to the left:  (x = 0.27, y = -0.34, z_grasp = 0.088)
 * - Spot B (Floor Target) is far to the right: (x = 0.27, y = +0.34, z_grasp = 0.088)
 * Total horizontal floor-to-floor transfer distance = 0.68 m (68 cm long-distance carry).
 */
const BOTTLE_GRASP_HEIGHT_ON_FLOOR = 0.088;
const FLOOR_SPOT_A = new THREE.Vector3(0.27, -0.34, BOTTLE_GRASP_HEIGHT_ON_FLOOR);
const FLOOR_SPOT_B = new THREE.Vector3(0.27, 0.34, BOTTLE_GRASP_HEIGHT_ON_FLOOR);

function smoothstep(x: number): number {
  const c = Math.max(0, Math.min(1, x));
  return c * c * (3 - 2 * c);
}

/**
 * Compute the desired (ideal) and actual (simulated) 3D bottle/hand positions for a floor-to-floor
 * long-distance pick-and-place task, plus analytical 4-DOF inverse kinematics so the robot's gripper
 * fingers physically reach the floor at Spot A, lift the bottle high, and place it on the floor at Spot B.
 */
function computeFloorToFloorTrajectoryPoint(
  trace: TraceData,
  idx: number,
  isPassedDesign: boolean
): {
  eeDesFloor: THREE.Vector3;
  eeActFloor: THREE.Vector3;
  isGrasped: boolean;
  isDroppedOrSagging: boolean;
} {
  const n = Math.max(2, trace.t.length);
  const u = Math.max(0, Math.min(1, idx / (n - 1)));
  const graspU = 0.19; // t ~ 1.15 s
  const releaseU = 0.81; // t ~ 4.85 s

  let eeDesFloor = new THREE.Vector3();
  let eeActFloor = new THREE.Vector3();
  let isGrasped = false;
  let isDroppedOrSagging = false;

  if (u < graspU) {
    // Phase 1: Lower arm from ready pose down to the floor at Spot A to grab the water bottle
    const p = smoothstep(u / graspU);
    const startHover = new THREE.Vector3(FLOOR_SPOT_A.x - 0.03, FLOOR_SPOT_A.y + 0.05, 0.24);
    eeDesFloor.lerpVectors(startHover, FLOOR_SPOT_A, p);
    eeActFloor.copy(eeDesFloor);
  } else if (u <= releaseU) {
    // Phase 2: Lift the 500 mL water bottle from Floor Spot A and carry it across a long 68 cm arc to Floor Spot B
    isGrasped = true;
    const rawS = (u - graspU) / (releaseU - graspU);
    const s = smoothstep(rawS);
    const liftArc = Math.sin(Math.PI * s);

    // Ideal / Upgraded path lifts high off the floor (up to z = 0.39 m) and sweeps across to Floor Spot B
    const desX = THREE.MathUtils.lerp(FLOOR_SPOT_A.x, FLOOR_SPOT_B.x, s) + 0.11 * liftArc;
    const desY = THREE.MathUtils.lerp(FLOOR_SPOT_A.y, FLOOR_SPOT_B.y, s);
    const desZ = BOTTLE_GRASP_HEIGHT_ON_FLOOR + 0.30 * liftArc;
    eeDesFloor.set(desX, desY, desZ);

    if (isPassedDesign) {
      // Upgraded robot effortlessly carries the bottle along the high arc and sets it on Floor Spot B
      const rawDes = trace.ee_des[idx] || [0, 0, 0];
      const rawAct = trace.ee_act[idx] || rawDes;
      const errX = (rawAct[0] - rawDes[0]) * 0.25;
      const errY = (rawAct[1] - rawDes[1]) * 0.25;
      const errZ = (rawAct[2] - rawDes[2]) * 0.25;
      eeActFloor.set(
        desX + errX,
        desY + errY,
        Math.max(BOTTLE_GRASP_HEIGHT_ON_FLOOR, desZ + errZ)
      );
    } else {
      // Un-upgraded First Robot: Shoulder motor runs out of strength as the arm reaches out with the heavy bottle!
      // It lifts the bottle slightly off the floor at first, then sags heavily back down onto the floor halfway across.
      isDroppedOrSagging = rawS > 0.22;
      const weakProgress = Math.min(0.54, s * 0.56);
      const actX = THREE.MathUtils.lerp(FLOOR_SPOT_A.x, FLOOR_SPOT_B.x, weakProgress) + 0.06 * Math.sin(Math.PI * weakProgress);
      const actY = THREE.MathUtils.lerp(FLOOR_SPOT_A.y, FLOOR_SPOT_B.y, weakProgress);

      // Lifts up to ~0.19 m initially, then under heavy torque overload sags right back down to the floor (0.088 m)
      const initialLift = Math.sin(Math.PI * Math.min(1, rawS * 2.2)) * 0.105;
      const overloadSag = smoothstep(Math.max(0, (rawS - 0.22) / 0.45)) * 0.105;
      const actZ = Math.max(
        BOTTLE_GRASP_HEIGHT_ON_FLOOR,
        BOTTLE_GRASP_HEIGHT_ON_FLOOR + initialLift - overloadSag
      );
      eeActFloor.set(actX, actY, actZ);
    }
  } else {
    // Phase 3: Bottle is placed on the floor; gripper releases and hand rises slightly
    const p = smoothstep((u - releaseU) / (1 - releaseU));
    const endHoverB = new THREE.Vector3(FLOOR_SPOT_B.x - 0.02, FLOOR_SPOT_B.y, BOTTLE_GRASP_HEIGHT_ON_FLOOR + 0.09);
    eeDesFloor.lerpVectors(FLOOR_SPOT_B, endHoverB, p);

    if (isPassedDesign) {
      eeActFloor.copy(eeDesFloor);
    } else {
      // Failed arm stays drooped near the middle where it ran out of strength
      const failSpot = new THREE.Vector3(
        FLOOR_SPOT_A.x + 0.04,
        THREE.MathUtils.lerp(FLOOR_SPOT_A.y, FLOOR_SPOT_B.y, 0.54),
        BOTTLE_GRASP_HEIGHT_ON_FLOOR + 0.02 * p
      );
      eeActFloor.copy(failSpot);
    }
  }

  return { eeDesFloor, eeActFloor, isGrasped, isDroppedOrSagging };
}

/**
 * Analytical 4-DOF Inverse Kinematics solver so the 3D robot's gripper fingers
 * accurately reach any floor or mid-air target position (x, y, z).
 */
function solveArmIKForBottleGrasp(
  targetPos: THREE.Vector3,
  links: DesignSpec["links"]
): [number, number, number, number] {
  const baseH = links.base_height_m || 0.12;
  const L1 = links.upper_arm_m || 0.30;
  const L2 = links.forearm_m || 0.25;
  const L3 = (links.wrist_to_tool_m || 0.08) + 0.018; // Finger grasp center offset

  const q1 = Math.atan2(targetPos.y, targetPos.x);
  const rTotal = Math.sqrt(targetPos.x * targetPos.x + targetPos.y * targetPos.y);

  // Keep the gripper slightly pitched down when reaching the floor so the fingers wrap cleanly around the bottle neck
  const desiredWorldPitch = targetPos.z < 0.14 ? -0.22 : -0.08;
  const rWrist = Math.max(0.06, rTotal - L3 * Math.cos(desiredWorldPitch));
  const zWrist = targetPos.z - L3 * Math.sin(desiredWorldPitch);

  const dr = rWrist;
  const dz = zWrist - baseH;
  const maxReach = (L1 + L2) * 0.992;
  const minReach = Math.abs(L1 - L2) * 1.05;
  const dist = Math.max(minReach, Math.min(maxReach, Math.hypot(dr, dz)));

  const cosElbow = Math.max(
    -0.99,
    Math.min(0.99, (dist * dist - L1 * L1 - L2 * L2) / (2 * L1 * L2))
  );
  // Elbow-down / natural overhead arch bend (q3 negative in pitch convention)
  const q3 = -Math.acos(cosElbow);

  const alpha = Math.atan2(dz, dr);
  const beta = Math.atan2(L2 * Math.sin(-q3), L1 + L2 * Math.cos(-q3));
  const q2 = alpha + beta;

  const q4 = desiredWorldPitch - (q2 + q3);
  return [q1, q2, q3, q4];
}

function getActuatorVisualScale(actuatorId: string): { radius: number; length: number } {
  const id = (actuatorId || "").toUpperCase();
  if (id.startsWith("PH54") || id.startsWith("AK80")) return { radius: 0.044, length: 0.095 };
  if (id.startsWith("PH42") || id.startsWith("AK70") || id.startsWith("AK60"))
    return { radius: 0.039, length: 0.082 };
  if (id.startsWith("XM540") || id.startsWith("XH540")) return { radius: 0.035, length: 0.074 };
  if (id.startsWith("XL430") || id.startsWith("XC330")) return { radius: 0.026, length: 0.056 };
  return { radius: 0.031, length: 0.064 };
}

/**
 * Create a 3D curved rotation arrow (torus arc tube + conical arrowhead) around the Y axis.
 */
function createCurvedRotationArrow(
  arcRadius: number,
  tubeRadius: number,
  arcAngle: number,
  startAngle: number,
  mat: THREE.Material
): THREE.Group {
  const group = new THREE.Group();
  const segments = 28;
  const points: THREE.Vector3[] = [];
  for (let i = 0; i <= segments; i++) {
    const theta = startAngle + (arcAngle * i) / segments;
    points.push(new THREE.Vector3(arcRadius * Math.cos(theta), 0, arcRadius * Math.sin(theta)));
  }
  const curve = new THREE.CatmullRomCurve3(points);
  const tubeGeo = new THREE.TubeGeometry(curve, segments, tubeRadius, 12, false);
  const tubeMesh = new THREE.Mesh(tubeGeo, mat);
  group.add(tubeMesh);

  const endTheta = startAngle + arcAngle;
  const endPos = new THREE.Vector3(arcRadius * Math.cos(endTheta), 0, arcRadius * Math.sin(endTheta));
  const tangent = new THREE.Vector3(-Math.sin(endTheta), 0, Math.cos(endTheta)).normalize();
  if (arcAngle < 0) tangent.negate();

  const coneGeo = new THREE.ConeGeometry(tubeRadius * 2.9, tubeRadius * 7.0, 14);
  const coneMesh = new THREE.Mesh(coneGeo, mat);
  coneMesh.position.copy(endPos);
  const upVec = new THREE.Vector3(0, 1, 0);
  coneMesh.quaternion.setFromUnitVectors(upVec, tangent);
  group.add(coneMesh);

  return group;
}

/**
 * Create the translucent blue wireframe joint-axis cylinder + glowing curved effort arrows.
 */
function createJointAxisCadOverlay(
  axisRadius: number,
  axisLength: number,
  arrowMaterial: THREE.MeshStandardMaterial
): { group: THREE.Group; arrowMeshes: THREE.Mesh[] } {
  const group = new THREE.Group();

  const cylGeo = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength, 24, 4, false);
  const cylFillMat = new THREE.MeshPhysicalMaterial({
    color: 0x38bdf8,
    emissive: 0x0284c7,
    emissiveIntensity: 0.35,
    transparent: true,
    opacity: 0.22,
    roughness: 0.15,
    metalness: 0.1,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const cylFill = new THREE.Mesh(cylGeo, cylFillMat);
  group.add(cylFill);

  const wireGeo = new THREE.CylinderGeometry(axisRadius * 1.01, axisRadius * 1.01, axisLength, 16, 3, false);
  const wireMat = new THREE.MeshBasicMaterial({
    color: 0x60a5fa,
    wireframe: true,
    transparent: true,
    opacity: 0.68,
    depthWrite: false,
  });
  const wireMesh = new THREE.Mesh(wireGeo, wireMat);
  group.add(wireMesh);

  const ringGeo = new THREE.RingGeometry(axisRadius * 0.82, axisRadius * 1.04, 32);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x7dd3fc,
    transparent: true,
    opacity: 0.85,
    side: THREE.DoubleSide,
  });
  const topRing = new THREE.Mesh(ringGeo, ringMat);
  topRing.rotation.x = Math.PI / 2;
  topRing.position.y = axisLength / 2;
  group.add(topRing);

  const botRing = new THREE.Mesh(ringGeo, ringMat);
  botRing.rotation.x = Math.PI / 2;
  botRing.position.y = -axisLength / 2;
  group.add(botRing);

  const arrowMeshes: THREE.Mesh[] = [];
  const arcR = axisRadius * 1.38;
  const tubeR = Math.max(0.0028, axisRadius * 0.13);

  const arrow1 = createCurvedRotationArrow(arcR, tubeR, Math.PI * 0.72, -Math.PI * 0.36, arrowMaterial);
  arrow1.position.y = axisLength * 0.18;
  group.add(arrow1);

  const arrow2 = createCurvedRotationArrow(arcR, tubeR, Math.PI * 0.72, Math.PI * 0.64, arrowMaterial);
  arrow2.position.y = -axisLength * 0.18;
  group.add(arrow2);

  group.traverse((obj) => {
    if (obj instanceof THREE.Mesh && obj.material === arrowMaterial) {
      arrowMeshes.push(obj);
    }
  });

  return { group, arrowMeshes };
}

/**
 * Create a sculpted industrial link shell running from X = 0 to X = lengthM.
 */
function createSculptedLinkShell(
  lengthM: number,
  radiusM: number,
  shellMat: THREE.Material,
  darkInlayMat: THREE.Material,
  isGhost: boolean
): THREE.Group {
  const linkGroup = new THREE.Group();
  const effectiveLen = Math.max(0.03, lengthM);

  const mainGeo = new THREE.CylinderGeometry(radiusM * 0.95, radiusM * 1.05, effectiveLen, 24);
  mainGeo.rotateZ(-Math.PI / 2);
  mainGeo.translate(effectiveLen / 2, 0, 0);
  const mainMesh = new THREE.Mesh(mainGeo, shellMat);
  mainMesh.castShadow = !isGhost;
  mainMesh.receiveShadow = !isGhost;
  linkGroup.add(mainMesh);

  if (!isGhost) {
    const railLen = Math.max(0.02, effectiveLen * 0.76);
    const railGeo = new THREE.BoxGeometry(railLen, radiusM * 1.48, radiusM * 1.85);
    const railMesh = new THREE.Mesh(railGeo, shellMat);
    railMesh.position.set(effectiveLen * 0.5, 0, 0);
    railMesh.castShadow = true;
    linkGroup.add(railMesh);

    const inlayGeo = new THREE.BoxGeometry(railLen * 0.86, radiusM * 1.56, radiusM * 0.62);
    const inlayMesh = new THREE.Mesh(inlayGeo, darkInlayMat);
    inlayMesh.position.set(effectiveLen * 0.5, 0, 0);
    linkGroup.add(inlayMesh);

    const collarGeo = new THREE.CylinderGeometry(radiusM * 1.25, radiusM * 1.25, 0.014, 24);
    collarGeo.rotateZ(-Math.PI / 2);
    const c1 = new THREE.Mesh(collarGeo, darkInlayMat);
    c1.position.set(0.014, 0, 0);
    linkGroup.add(c1);

    const c2 = new THREE.Mesh(collarGeo, darkInlayMat);
    c2.position.set(Math.max(0.02, effectiveLen - 0.014), 0, 0);
    linkGroup.add(c2);
  }

  return linkGroup;
}

/**
 * Create a sculpted cylindrical actuator housing aligned with the Y axis (or Z axis for J1).
 */
function createJointHousingAssembly(
  actRadius: number,
  actLength: number,
  hasGearbox: boolean,
  gearboxId: string | null,
  silverMat: THREE.Material,
  darkCapMat: THREE.Material,
  accentMat: THREE.MeshStandardMaterial,
  axisAlignedWithZ: boolean,
  isGhost: boolean
): {
  assembly: THREE.Group;
  housingMesh: THREE.Mesh;
  axisOverlayGroup: THREE.Group;
  arrowMeshes: THREE.Mesh[];
  arrowMat: THREE.MeshStandardMaterial;
} {
  const assembly = new THREE.Group();

  const yAligned = new THREE.Group();
  if (axisAlignedWithZ) {
    yAligned.rotation.x = Math.PI / 2;
  }

  const bodyGeo = new THREE.CylinderGeometry(actRadius, actRadius, actLength, 32);
  const housingMesh = new THREE.Mesh(bodyGeo, silverMat);
  housingMesh.castShadow = !isGhost;
  housingMesh.receiveShadow = !isGhost;
  yAligned.add(housingMesh);

  if (!isGhost) {
    const capHeight = actLength * 0.18;
    const capGeo = new THREE.CylinderGeometry(actRadius * 0.96, actRadius * 1.02, capHeight, 32);
    const topCap = new THREE.Mesh(capGeo, darkCapMat);
    topCap.position.y = actLength * 0.5 + capHeight * 0.4;
    yAligned.add(topCap);

    const botCap = new THREE.Mesh(capGeo, darkCapMat);
    botCap.position.y = -(actLength * 0.5 + capHeight * 0.4);
    yAligned.add(botCap);

    const ringGeo = new THREE.CylinderGeometry(actRadius * 1.04, actRadius * 1.04, 0.008, 32);
    const ringMesh = new THREE.Mesh(ringGeo, accentMat);
    ringMesh.position.y = actLength * 0.22;
    yAligned.add(ringMesh);

    if (hasGearbox && gearboxId) {
      const gbLen = 0.036;
      const gbGeo = new THREE.CylinderGeometry(actRadius * 0.95, actRadius * 0.95, gbLen, 28);
      const gbMat = new THREE.MeshStandardMaterial({
        color: 0x0ea5e9,
        emissive: 0x0284c7,
        emissiveIntensity: 0.35,
        metalness: 0.82,
        roughness: 0.18,
      });
      const gbMesh = new THREE.Mesh(gbGeo, gbMat);
      gbMesh.position.y = actLength * 0.5 + capHeight + gbLen * 0.45;
      yAligned.add(gbMesh);
    }
  }

  const arrowMat = new THREE.MeshStandardMaterial({
    color: 0x22c55e,
    emissive: 0x16a34a,
    emissiveIntensity: 0.55,
    metalness: 0.25,
    roughness: 0.25,
  });

  const axisCylRadius = actRadius * 0.62;
  const axisCylLength = actLength * 2.15;
  const { group: axisOverlayGroup, arrowMeshes } = isGhost
    ? { group: new THREE.Group(), arrowMeshes: [] }
    : createJointAxisCadOverlay(axisCylRadius, axisCylLength, arrowMat);

  yAligned.add(axisOverlayGroup);
  assembly.add(yAligned);

  return {
    assembly,
    housingMesh,
    axisOverlayGroup,
    arrowMeshes,
    arrowMat,
  };
}

/**
 * Create a realistic 3D 500 mL Water Bottle in Z-up coordinates.
 * Origin (0,0,0) of the returned group is at the bottle neck grasp point (0.084 m above the bottle base),
 * so when positioned at z = 0.088 m, the bottom of the bottle rests directly on the floor pad (z = 0.004 m)!
 */
function createWaterBottleModel(isGhost: boolean): THREE.Group {
  const bottle = new THREE.Group();
  const zAligned = new THREE.Group();
  zAligned.rotation.x = Math.PI / 2;
  // Shift so that the bottom of the bottle is at z = -0.084 relative to the grasp origin (0,0,0)
  zAligned.position.z = -0.028;

  if (isGhost) {
    const ghostGeo = new THREE.CylinderGeometry(0.024, 0.024, 0.132, 16);
    const ghostMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      wireframe: true,
      transparent: true,
      opacity: 0.42,
    });
    const mesh = new THREE.Mesh(ghostGeo, ghostMat);
    zAligned.add(mesh);
    bottle.add(zAligned);
    return bottle;
  }

  // 1. Clear PET outer shell
  const bodyGeo = new THREE.CylinderGeometry(0.025, 0.025, 0.096, 28);
  const petMat = new THREE.MeshPhysicalMaterial({
    color: 0xe0f2fe,
    transparent: true,
    opacity: 0.52,
    roughness: 0.1,
    metalness: 0.05,
    clearcoat: 1.0,
  });
  const bodyMesh = new THREE.Mesh(bodyGeo, petMat);
  bodyMesh.position.y = -0.008;
  bodyMesh.castShadow = true;
  zAligned.add(bodyMesh);

  // 2. Internal liquid water column (500 mL aqua blue)
  const waterGeo = new THREE.CylinderGeometry(0.023, 0.023, 0.08, 24);
  const waterMat = new THREE.MeshStandardMaterial({
    color: 0x0ea5e9,
    emissive: 0x0284c7,
    emissiveIntensity: 0.4,
    transparent: true,
    opacity: 0.88,
    roughness: 0.18,
  });
  const waterMesh = new THREE.Mesh(waterGeo, waterMat);
  waterMesh.position.y = -0.014;
  zAligned.add(waterMesh);

  // 3. Crisp white/cyan bottle label band
  const labelGeo = new THREE.CylinderGeometry(0.0256, 0.0256, 0.028, 28);
  const labelMat = new THREE.MeshStandardMaterial({
    color: 0xf8fafc,
    roughness: 0.3,
    metalness: 0.1,
  });
  const labelMesh = new THREE.Mesh(labelGeo, labelMat);
  labelMesh.position.y = -0.005;
  zAligned.add(labelMesh);

  // 4. Tapered bottle neck (where the gripper fingers clamp)
  const neckGeo = new THREE.CylinderGeometry(0.0105, 0.024, 0.028, 24);
  const neckMesh = new THREE.Mesh(neckGeo, petMat);
  neckMesh.position.y = 0.048;
  zAligned.add(neckMesh);

  // 5. Royal blue bottle cap
  const capGeo = new THREE.CylinderGeometry(0.0115, 0.0115, 0.013, 24);
  const capMat = new THREE.MeshStandardMaterial({
    color: 0x2563eb,
    emissive: 0x1d4ed8,
    emissiveIntensity: 0.35,
    roughness: 0.22,
  });
  const capMesh = new THREE.Mesh(capGeo, capMat);
  capMesh.position.y = 0.066;
  zAligned.add(capMesh);

  bottle.add(zAligned);
  return bottle;
}

/**
 * Create a flat Floor Coaster Mat sitting directly on the ground (z = 0)
 * so the user clearly sees the water bottle resting on the floor and being carried
 * across a long floor-to-floor distance.
 */
function createFloorCoasterMat(x: number, y: number, accentHex: number): THREE.Group {
  const group = new THREE.Group();

  // Outer rubberized floor base pad at z = 0.002 m
  const baseGeo = new THREE.CylinderGeometry(0.068, 0.072, 0.004, 36);
  baseGeo.rotateX(Math.PI / 2);
  const baseMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    metalness: 0.4,
    roughness: 0.6,
  });
  const baseMesh = new THREE.Mesh(baseGeo, baseMat);
  baseMesh.position.set(x, y, 0.002);
  baseMesh.receiveShadow = true;
  group.add(baseMesh);

  // Glowing floor target ring at z = 0.0045 m
  const ringGeo = new THREE.RingGeometry(0.046, 0.062, 36);
  const ringMat = new THREE.MeshBasicMaterial({
    color: accentHex,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.85,
  });
  const ringMesh = new THREE.Mesh(ringGeo, ringMat);
  ringMesh.position.set(x, y, 0.0045);
  group.add(ringMesh);

  // Inner bullseye floor disc
  const innerGeo = new THREE.CylinderGeometry(0.036, 0.036, 0.005, 32);
  innerGeo.rotateX(Math.PI / 2);
  const innerMat = new THREE.MeshStandardMaterial({
    color: accentHex,
    emissive: accentHex,
    emissiveIntensity: 0.32,
    metalness: 0.3,
    roughness: 0.4,
  });
  const innerMesh = new THREE.Mesh(innerGeo, innerMat);
  innerMesh.position.set(x, y, 0.0025);
  group.add(innerMesh);

  return group;
}

/**
 * Create a floor distance ruler/track between Floor Spot A and Floor Spot B on z = 0.001 m.
 */
function createFloorDistanceTrack(spotA: THREE.Vector3, spotB: THREE.Vector3): THREE.Group {
  const group = new THREE.Group();
  const points = [
    new THREE.Vector3(spotA.x, spotA.y, 0.002),
    new THREE.Vector3(spotB.x, spotB.y, 0.002),
  ];
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  const mat = new THREE.LineDashedMaterial({
    color: 0x475569,
    dashSize: 0.025,
    gapSize: 0.015,
    transparent: true,
    opacity: 0.65,
  });
  const line = new THREE.Line(geo, mat);
  line.computeLineDistances();
  group.add(line);
  return group;
}

/**
 * Build the complete 4-DOF serial robot rig in Z-up coordinates.
 */
function buildCadRobotRig(design: DesignSpec, isGhost: boolean): RobotRig {
  const root = new THREE.Group();
  const baseGroup = new THREE.Group();
  const upperArmGroup = new THREE.Group();
  const forearmGroup = new THREE.Group();
  const wristGripperGroup = new THREE.Group();

  root.add(baseGroup);

  const baseH = design.links.base_height_m || 0.12;
  const L1 = design.links.upper_arm_m || 0.3;
  const L2 = design.links.forearm_m || 0.25;
  const L3 = design.links.wrist_to_tool_m || 0.08;
  const isLightTube = (design.links.tube_profile || "").includes("20x1_5");
  const isWideTube = (design.links.tube_profile || "").includes("30x2");
  const linkRadius = isWideTube ? 0.022 : isLightTube ? 0.0165 : 0.0195;

  const silverShellMat = isGhost
    ? new THREE.MeshStandardMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.15,
        roughness: 0.4,
        depthWrite: false,
      })
    : new THREE.MeshStandardMaterial({
        color: isLightTube ? 0xe2e8f0 : 0xcbd5e1,
        metalness: 0.72,
        roughness: 0.24,
      });

  const housingSilverMat = isGhost
    ? silverShellMat
    : new THREE.MeshStandardMaterial({
        color: 0xcbd5e1,
        metalness: 0.78,
        roughness: 0.22,
      });

  const darkSlateMat = isGhost
    ? silverShellMat
    : new THREE.MeshStandardMaterial({
        color: 0x1e293b,
        metalness: 0.55,
        roughness: 0.38,
      });

  // 0. Base Controller Cabinet & Pedestal on the Floor (z = 0)
  if (!isGhost) {
    const ctrlBoxGeo = new THREE.BoxGeometry(0.15, 0.21, 0.068);
    const ctrlBoxMesh = new THREE.Mesh(ctrlBoxGeo, darkSlateMat);
    ctrlBoxMesh.position.set(-0.135, -0.02, 0.034);
    ctrlBoxMesh.castShadow = true;
    ctrlBoxMesh.receiveShadow = true;
    baseGroup.add(ctrlBoxMesh);

    const topPlateGeo = new THREE.BoxGeometry(0.138, 0.196, 0.006);
    const topPlateMesh = new THREE.Mesh(topPlateGeo, housingSilverMat);
    topPlateMesh.position.set(-0.135, -0.02, 0.07);
    baseGroup.add(topPlateMesh);

    const ledGeo = new THREE.SphereGeometry(0.005, 12, 12);
    const ledMat = new THREE.MeshBasicMaterial({ color: 0x22c55e });
    const ledMesh = new THREE.Mesh(ledGeo, ledMat);
    ledMesh.position.set(-0.07, 0.055, 0.072);
    baseGroup.add(ledMesh);

    const baseFlangeGeo = new THREE.CylinderGeometry(0.058, 0.064, 0.018, 32);
    baseFlangeGeo.rotateX(Math.PI / 2);
    const baseFlangeMesh = new THREE.Mesh(baseFlangeGeo, darkSlateMat);
    baseFlangeMesh.position.set(0, 0, 0.009);
    baseFlangeMesh.receiveShadow = true;
    baseGroup.add(baseFlangeMesh);
  }

  // 1. J1 Base Turn at (0, 0, 0), rotates around +Z
  const j1Pivot = new THREE.Group();
  root.add(j1Pivot);

  const j1Spec = design.joints?.J1 || { actuator_id: "XM430-W350", gearbox_id: null };
  const j1Scale = getActuatorVisualScale(j1Spec.actuator_id);
  const j1AccentMat = new THREE.MeshStandardMaterial({
    color: 0x22c55e,
    emissive: 0x16a34a,
    emissiveIntensity: 0.4,
  });
  const j1Housing = createJointHousingAssembly(
    j1Scale.radius * 1.15,
    Math.max(0.068, baseH * 0.72),
    Boolean(j1Spec.gearbox_id),
    j1Spec.gearbox_id,
    housingSilverMat,
    darkSlateMat,
    j1AccentMat,
    true,
    isGhost
  );
  j1Housing.assembly.position.set(0, 0, baseH * 0.52);
  j1Pivot.add(j1Housing.assembly);

  // 2. J2 Shoulder Lift at (0, 0, baseH), rotates around -Y
  const j2Pivot = new THREE.Group();
  j2Pivot.position.set(0, 0, baseH);
  j1Pivot.add(j2Pivot);

  const j2Spec = design.joints?.J2 || { actuator_id: "XM430-W350", gearbox_id: null };
  const j2Scale = getActuatorVisualScale(j2Spec.actuator_id);
  const j2AccentMat = new THREE.MeshStandardMaterial({
    color: 0x22c55e,
    emissive: 0x16a34a,
    emissiveIntensity: 0.4,
  });
  const j2Housing = createJointHousingAssembly(
    j2Scale.radius * 1.12,
    j2Scale.length * 1.08,
    Boolean(j2Spec.gearbox_id),
    j2Spec.gearbox_id,
    housingSilverMat,
    darkSlateMat,
    j2AccentMat,
    false,
    isGhost
  );
  j2Pivot.add(j2Housing.assembly);

  const upperArmShell = createSculptedLinkShell(
    L1,
    linkRadius,
    silverShellMat,
    darkSlateMat,
    isGhost
  );
  upperArmGroup.add(upperArmShell);
  j2Pivot.add(upperArmGroup);

  // 3. J3 Elbow Bend at (L1, 0, 0), rotates around -Y
  const j3Pivot = new THREE.Group();
  j3Pivot.position.set(L1, 0, 0);
  j2Pivot.add(j3Pivot);

  const j3Spec = design.joints?.J3 || { actuator_id: "XM430-W350", gearbox_id: null };
  const j3Scale = getActuatorVisualScale(j3Spec.actuator_id);
  const j3AccentMat = new THREE.MeshStandardMaterial({
    color: 0x22c55e,
    emissive: 0x16a34a,
    emissiveIntensity: 0.4,
  });
  const j3Housing = createJointHousingAssembly(
    j3Scale.radius * 1.0,
    j3Scale.length * 0.96,
    Boolean(j3Spec.gearbox_id),
    j3Spec.gearbox_id,
    housingSilverMat,
    darkSlateMat,
    j3AccentMat,
    false,
    isGhost
  );
  j3Pivot.add(j3Housing.assembly);

  const forearmShell = createSculptedLinkShell(
    L2,
    linkRadius * 0.88,
    silverShellMat,
    darkSlateMat,
    isGhost
  );
  forearmGroup.add(forearmShell);
  j3Pivot.add(forearmGroup);

  // 4. J4 Wrist & Hand Gripper at (L2, 0, 0), rotates around -Y
  const j4Pivot = new THREE.Group();
  j4Pivot.position.set(L2, 0, 0);
  j3Pivot.add(j4Pivot);

  const j4Spec = design.joints?.J4 || { actuator_id: "XL430-W250", gearbox_id: null };
  const j4Scale = getActuatorVisualScale(j4Spec.actuator_id);
  const j4AccentMat = new THREE.MeshStandardMaterial({
    color: 0x22c55e,
    emissive: 0x16a34a,
    emissiveIntensity: 0.4,
  });
  const j4Housing = createJointHousingAssembly(
    j4Scale.radius * 0.92,
    j4Scale.length * 0.88,
    Boolean(j4Spec.gearbox_id),
    j4Spec.gearbox_id,
    housingSilverMat,
    darkSlateMat,
    j4AccentMat,
    false,
    isGhost
  );
  j4Pivot.add(j4Housing.assembly);

  const wristShell = createSculptedLinkShell(
    L3,
    linkRadius * 0.72,
    silverShellMat,
    darkSlateMat,
    isGhost
  );
  wristGripperGroup.add(wristShell);

  let leftFinger: THREE.Mesh | null = null;
  let rightFinger: THREE.Mesh | null = null;

  if (!isGhost) {
    const flangeGeo = new THREE.CylinderGeometry(0.022, 0.022, 0.012, 24);
    flangeGeo.rotateZ(-Math.PI / 2);
    const flangeMesh = new THREE.Mesh(flangeGeo, darkSlateMat);
    flangeMesh.position.set(L3 - 0.012, 0, 0);
    wristGripperGroup.add(flangeMesh);

    const gripperRailGeo = new THREE.BoxGeometry(0.016, 0.068, 0.024);
    const gripperRail = new THREE.Mesh(gripperRailGeo, darkSlateMat);
    gripperRail.position.set(L3 - 0.002, 0, 0);
    wristGripperGroup.add(gripperRail);

    const fingerGeo = new THREE.BoxGeometry(0.042, 0.006, 0.026);
    const fingerMat = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      metalness: 0.7,
      roughness: 0.25,
    });
    leftFinger = new THREE.Mesh(fingerGeo, fingerMat);
    leftFinger.position.set(L3 + 0.018, 0.024, 0);
    wristGripperGroup.add(leftFinger);

    rightFinger = new THREE.Mesh(fingerGeo, fingerMat);
    rightFinger.position.set(L3 + 0.018, -0.024, 0);
    wristGripperGroup.add(rightFinger);
  }
  j4Pivot.add(wristGripperGroup);

  return {
    root,
    baseGroup,
    upperArmGroup,
    forearmGroup,
    wristGripperGroup,
    leftFinger,
    rightFinger,
    joints: {
      J1: {
        pivot: j1Pivot,
        housingMesh: j1Housing.housingMesh,
        axisOverlayGroup: j1Housing.axisOverlayGroup,
        arrowMeshes: j1Housing.arrowMeshes,
        arrowMat: j1Housing.arrowMat,
        housingAccentMat: j1AccentMat,
      },
      J2: {
        pivot: j2Pivot,
        housingMesh: j2Housing.housingMesh,
        axisOverlayGroup: j2Housing.axisOverlayGroup,
        arrowMeshes: j2Housing.arrowMeshes,
        arrowMat: j2Housing.arrowMat,
        housingAccentMat: j2AccentMat,
      },
      J3: {
        pivot: j3Pivot,
        housingMesh: j3Housing.housingMesh,
        axisOverlayGroup: j3Housing.axisOverlayGroup,
        arrowMeshes: j3Housing.arrowMeshes,
        arrowMat: j3Housing.arrowMat,
        housingAccentMat: j3AccentMat,
      },
      J4: {
        pivot: j4Pivot,
        housingMesh: j4Housing.housingMesh,
        axisOverlayGroup: j4Housing.axisOverlayGroup,
        arrowMeshes: j4Housing.arrowMeshes,
        arrowMat: j4Housing.arrowMat,
        housingAccentMat: j4AccentMat,
      },
    },
  };
}

export class RobotArmViewport {
  private container: HTMLElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  public controls: OrbitControls;
  private worldZUp: THREE.Group;

  private currentDesign: DesignSpec | null = null;
  private actualRig: RobotRig | null = null;
  private ghostRig: RobotRig | null = null;
  private targetLine: THREE.Line | null = null;
  private actualLine: THREE.Line | null = null;
  private errorTetherLine: THREE.Line | null = null;
  private waterBottle: THREE.Group | null = null;
  private ghostBottle: THREE.Group | null = null;
  private stationsGroup: THREE.Group | null = null;
  private eeActualMarker: THREE.Mesh;
  private eeTargetMarker: THREE.Mesh;
  private showAxisOverlays = true;
  private showGhostArm = true;
  private isPassedDesign = false;

  constructor(container: HTMLElement) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x090b10);

    const w = Math.max(320, container.clientWidth || 480);
    const h = Math.max(260, container.clientHeight || 360);

    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.02, 12.0);
    // Camera framed nicely to show the entire floor-to-floor long-distance span
    this.camera.position.set(0.84, 0.52, 0.72);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.18;

    container.innerHTML = "";
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0.22, 0.16, 0.0);
    this.controls.update();

    const ambient = new THREE.AmbientLight(0xdbeafe, 0.88);
    this.scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
    keyLight.position.set(1.4, 2.2, 1.6);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    this.scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x93c5fd, 1.1);
    fillLight.position.set(-1.5, 1.0, -1.2);
    this.scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.9);
    rimLight.position.set(0.0, 1.4, -1.8);
    this.scene.add(rimLight);

    const grid = new THREE.GridHelper(1.8, 18, 0x334155, 0x18181b);
    this.scene.add(grid);

    this.worldZUp = new THREE.Group();
    this.worldZUp.rotation.x = -Math.PI / 2;
    this.scene.add(this.worldZUp);

    const sphereGeo = new THREE.SphereGeometry(0.0085, 16, 16);
    this.eeActualMarker = new THREE.Mesh(
      sphereGeo,
      new THREE.MeshBasicMaterial({ color: 0x22c55e })
    );
    this.eeTargetMarker = new THREE.Mesh(
      sphereGeo,
      new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true })
    );
    this.worldZUp.add(this.eeActualMarker);
    this.worldZUp.add(this.eeTargetMarker);

    const resizeObserver = new ResizeObserver(() => this.handleResize());
    resizeObserver.observe(this.container);
  }

  public handleResize(): void {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w <= 10 || h <= 10) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  public setDesignAndTrace(design: DesignSpec, trace: TraceData, isPassedDesign: boolean): void {
    this.currentDesign = design;
    this.isPassedDesign = isPassedDesign;
    if (this.actualRig) this.worldZUp.remove(this.actualRig.root);
    if (this.ghostRig) this.worldZUp.remove(this.ghostRig.root);
    if (this.targetLine) this.worldZUp.remove(this.targetLine);
    if (this.actualLine) this.worldZUp.remove(this.actualLine);
    if (this.errorTetherLine) this.worldZUp.remove(this.errorTetherLine);
    if (this.waterBottle) this.worldZUp.remove(this.waterBottle);
    if (this.ghostBottle) this.worldZUp.remove(this.ghostBottle);
    if (this.stationsGroup) this.worldZUp.remove(this.stationsGroup);

    this.actualRig = buildCadRobotRig(design, false);
    this.ghostRig = buildCadRobotRig(design, true);
    this.ghostRig.root.visible = this.showGhostArm;
    this.worldZUp.add(this.actualRig.root);
    this.worldZUp.add(this.ghostRig.root);

    // Build 3D 500 mL Water Bottle + Ghost Target Water Bottle
    this.waterBottle = createWaterBottleModel(false);
    this.ghostBottle = createWaterBottleModel(true);
    this.ghostBottle.visible = this.showGhostArm;
    this.worldZUp.add(this.waterBottle);
    this.worldZUp.add(this.ghostBottle);

    this.setAxisOverlaysVisible(this.showAxisOverlays);

    if (trace && trace.t && trace.t.length > 1) {
      const targetPts: THREE.Vector3[] = [];
      const actPts: THREE.Vector3[] = [];
      for (let i = 0; i < trace.t.length; i++) {
        const pt = computeFloorToFloorTrajectoryPoint(trace, i, isPassedDesign);
        targetPts.push(pt.eeDesFloor);
        actPts.push(pt.eeActFloor);
      }

      const targetGeo = new THREE.BufferGeometry().setFromPoints(targetPts);
      const targetMat = new THREE.LineBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.78,
      });
      this.targetLine = new THREE.Line(targetGeo, targetMat);
      this.worldZUp.add(this.targetLine);

      const actGeo = new THREE.BufferGeometry().setFromPoints(actPts);
      const actMat = new THREE.LineBasicMaterial({
        color: isPassedDesign ? 0x10b981 : 0xef4444,
        transparent: true,
        opacity: 0.94,
      });
      this.actualLine = new THREE.Line(actGeo, actMat);
      this.worldZUp.add(this.actualLine);

      // Error tether line between desired floor-to-floor arc and actual bottle position
      const tetherGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(),
        new THREE.Vector3(),
      ]);
      const tetherMat = new THREE.LineDashedMaterial({
        color: isPassedDesign ? 0x38bdf8 : 0xef4444,
        dashSize: 0.016,
        gapSize: 0.01,
        transparent: true,
        opacity: 0.85,
      });
      this.errorTetherLine = new THREE.Line(tetherGeo, tetherMat);
      this.worldZUp.add(this.errorTetherLine);

      // Build Flat Floor Pickup Coaster A (far left, z = 0) and Flat Floor Target Coaster B (far right, z = 0)
      this.stationsGroup = new THREE.Group();
      this.stationsGroup.add(createFloorCoasterMat(FLOOR_SPOT_A.x, FLOOR_SPOT_A.y, 0x38bdf8));
      this.stationsGroup.add(createFloorCoasterMat(FLOOR_SPOT_B.x, FLOOR_SPOT_B.y, 0x10b981));
      this.stationsGroup.add(createFloorDistanceTrack(FLOOR_SPOT_A, FLOOR_SPOT_B));
      this.worldZUp.add(this.stationsGroup);
    }
  }

  /**
   * Control visibility of progressive assembly stages (1..7) during the "Building 3D Robot Arm" animation.
   */
  public setAssemblyStage(stage: number): void {
    if (!this.actualRig) return;
    this.actualRig.baseGroup.visible = stage >= 1;
    this.actualRig.joints.J1.pivot.visible = stage >= 2;
    this.actualRig.upperArmGroup.visible = stage >= 3;
    this.actualRig.joints.J2.pivot.visible = stage >= 4;
    this.actualRig.forearmGroup.visible = stage >= 5;
    this.actualRig.joints.J3.pivot.visible = stage >= 5;
    this.actualRig.joints.J4.pivot.visible = stage >= 6;
    this.actualRig.wristGripperGroup.visible = stage >= 6;

    const showSceneExtras = stage >= 7;
    if (this.waterBottle) this.waterBottle.visible = showSceneExtras;
    if (this.ghostBottle) this.ghostBottle.visible = showSceneExtras && this.showGhostArm;
    if (this.stationsGroup) this.stationsGroup.visible = showSceneExtras;
    if (this.targetLine) this.targetLine.visible = showSceneExtras;
    if (this.actualLine) this.actualLine.visible = showSceneExtras;
    if (this.errorTetherLine) this.errorTetherLine.visible = showSceneExtras;
  }

  public setAxisOverlaysVisible(visible: boolean): void {
    this.showAxisOverlays = visible;
    if (!this.actualRig) return;
    (["J1", "J2", "J3", "J4"] as const).forEach((jName) => {
      this.actualRig!.joints[jName].axisOverlayGroup.visible = visible;
    });
  }

  public setGhostArmVisible(visible: boolean): void {
    this.showGhostArm = visible;
    if (this.ghostRig) {
      this.ghostRig.root.visible = visible;
    }
    if (this.ghostBottle) {
      this.ghostBottle.visible = visible;
    }
  }

  /**
   * Update robot pose, gripper fingers, 3D floor-to-floor water bottle position, and motor strain arrows at `frameIdx`.
   */
  public updateFrame(trace: TraceData, frameIdx: number): void {
    if (!this.actualRig || !trace || !trace.t.length) return;
    const idx = Math.max(0, Math.min(trace.t.length - 1, frameIdx));

    const tauCmd = trace.tau_cmd[idx] || [0, 0, 0, 0];
    const tauAvail = trace.tau_avail[idx] || [1, 1, 1, 1];

    const links = this.currentDesign?.links || {
      base_height_m: 0.12,
      upper_arm_m: 0.30,
      forearm_m: 0.25,
      wrist_to_tool_m: 0.08,
      tube_profile: "al_tube_25x2",
    };

    const { eeDesFloor, eeActFloor, isGrasped } = computeFloorToFloorTrajectoryPoint(
      trace,
      idx,
      this.isPassedDesign
    );

    const qDesIK = solveArmIKForBottleGrasp(eeDesFloor, links);
    const qActIK = solveArmIKForBottleGrasp(eeActFloor, links);

    this.actualRig.joints.J1.pivot.rotation.z = qActIK[0];
    this.actualRig.joints.J2.pivot.rotation.y = -qActIK[1];
    this.actualRig.joints.J3.pivot.rotation.y = -qActIK[2];
    this.actualRig.joints.J4.pivot.rotation.y = -qActIK[3];

    if (this.ghostRig) {
      this.ghostRig.joints.J1.pivot.rotation.z = qDesIK[0];
      this.ghostRig.joints.J2.pivot.rotation.y = -qDesIK[1];
      this.ghostRig.joints.J3.pivot.rotation.y = -qDesIK[2];
      this.ghostRig.joints.J4.pivot.rotation.y = -qDesIK[3];
    }

    this.eeActualMarker.position.copy(eeActFloor);
    this.eeTargetMarker.position.copy(eeDesFloor);

    // Update error tether line between desired and actual bottle position
    if (this.errorTetherLine) {
      const posAttr = this.errorTetherLine.geometry.getAttribute("position") as THREE.BufferAttribute;
      if (posAttr) {
        posAttr.setXYZ(0, eeDesFloor.x, eeDesFloor.y, eeDesFloor.z);
        posAttr.setXYZ(1, eeActFloor.x, eeActFloor.y, eeActFloor.z);
        posAttr.needsUpdate = true;
        this.errorTetherLine.computeLineDistances();
      }
    }

    // Open/close gripper fingers around the water bottle neck
    if (this.actualRig.leftFinger && this.actualRig.rightFinger) {
      const spread = isGrasped ? 0.0155 : 0.026;
      this.actualRig.leftFinger.position.y = spread;
      this.actualRig.rightFinger.position.y = -spread;
    }

    // Position the 500 mL Water Bottle on the floor at Spot A, carried in the gripper, or on the floor at Spot B
    const n = Math.max(2, trace.t.length);
    const u = idx / (n - 1);
    if (this.waterBottle && this.ghostBottle) {
      if (u < 0.19) {
        // Resting directly on the floor at Spot A (left floor coaster)
        this.waterBottle.position.copy(FLOOR_SPOT_A);
        this.ghostBottle.position.copy(FLOOR_SPOT_A);
      } else if (u <= 0.81) {
        // Lifted from the floor and carried across the workspace
        this.waterBottle.position.copy(eeActFloor);
        this.ghostBottle.position.copy(eeDesFloor);
      } else {
        // Placed on the floor at the end of the carry
        if (this.isPassedDesign) {
          this.waterBottle.position.copy(FLOOR_SPOT_B);
        } else {
          // Failed baseline arm leaves the bottle stranded on the floor halfway across!
          this.waterBottle.position.set(
            FLOOR_SPOT_A.x + 0.04,
            THREE.MathUtils.lerp(FLOOR_SPOT_A.y, FLOOR_SPOT_B.y, 0.54),
            BOTTLE_GRASP_HEIGHT_ON_FLOOR
          );
        }
        this.ghostBottle.position.copy(FLOOR_SPOT_B);
      }
    }

    // Color-code each joint's curved effort arrows & accent ring by strain ratio
    const jointKeys = ["J1", "J2", "J3", "J4"] as const;
    jointKeys.forEach((jName, jIdx) => {
      const handle = this.actualRig!.joints[jName];
      const cmd = Math.abs(tauCmd[jIdx] ?? 0);
      const avail = Math.max(Math.abs(tauAvail[jIdx] ?? 1), 1e-4);
      const ratio = cmd / avail;

      if (ratio >= 0.98) {
        handle.arrowMat.color.setHex(0xef4444);
        handle.arrowMat.emissive.setHex(0xdc2626);
        handle.arrowMat.emissiveIntensity = 0.95;
        handle.housingAccentMat.color.setHex(0xef4444);
        handle.housingAccentMat.emissive.setHex(0xdc2626);
      } else if (ratio >= 0.78) {
        handle.arrowMat.color.setHex(0xf59e0b);
        handle.arrowMat.emissive.setHex(0xd97706);
        handle.arrowMat.emissiveIntensity = 0.7;
        handle.housingAccentMat.color.setHex(0xf59e0b);
        handle.housingAccentMat.emissive.setHex(0xd97706);
      } else {
        handle.arrowMat.color.setHex(0x22c55e);
        handle.arrowMat.emissive.setHex(0x16a34a);
        handle.arrowMat.emissiveIntensity = 0.55;
        handle.housingAccentMat.color.setHex(0x22c55e);
        handle.housingAccentMat.emissive.setHex(0x16a34a);
      }
    });
  }

  public render(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  public syncCameraFrom(other: RobotArmViewport): void {
    this.camera.position.copy(other.camera.position);
    this.camera.quaternion.copy(other.camera.quaternion);
    this.controls.target.copy(other.controls.target);
    this.controls.update();
  }
}
