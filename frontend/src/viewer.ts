/**
 * CAD/CAE 3D Robot Arm & Water-Bottle Pick-and-Transfer Simulation Viewer (Three.js)
 * Models the 4-DOF serial robot arm with sculpted metallic/anodized joint housings,
 * industrial base controller cabinet, chamfered link shells, translucent blue wireframe
 * joint-axis cylinders, glowing green curved rotation/torque arrows, 2-finger parallel gripper,
 * Pickup/Target stations, and a 3D 500 mL water bottle payload.
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
 * Create the translucent blue wireframe joint-axis cylinder + glowing green curved torque arrows.
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
      const gbLen = 0.034;
      const gbGeo = new THREE.CylinderGeometry(actRadius * 0.92, actRadius * 0.92, gbLen, 28);
      const gbMat = new THREE.MeshStandardMaterial({
        color: 0x0ea5e9,
        emissive: 0x0284c7,
        emissiveIntensity: 0.28,
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
 * Origin (0,0,0) of the returned group is at the bottle neck grasp point so it aligns with ee_act / ee_des.
 */
function createWaterBottleModel(isGhost: boolean): THREE.Group {
  const bottle = new THREE.Group();
  // In Z-up, cylinder default is Y-aligned, so rotateX(Math.PI / 2) makes it vertical along Z.
  const zAligned = new THREE.Group();
  zAligned.rotation.x = Math.PI / 2;
  // Shift down slightly so origin (0,0,0) is around the upper neck/shoulder grasp zone
  zAligned.position.z = -0.032;

  if (isGhost) {
    const ghostGeo = new THREE.CylinderGeometry(0.022, 0.022, 0.125, 16);
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
  const bodyGeo = new THREE.CylinderGeometry(0.023, 0.023, 0.088, 28);
  const petMat = new THREE.MeshPhysicalMaterial({
    color: 0xe0f2fe,
    transparent: true,
    opacity: 0.48,
    roughness: 0.12,
    metalness: 0.05,
    clearcoat: 1.0,
  });
  const bodyMesh = new THREE.Mesh(bodyGeo, petMat);
  bodyMesh.position.y = -0.012;
  bodyMesh.castShadow = true;
  zAligned.add(bodyMesh);

  // 2. Internal liquid water column (500 mL aqua blue)
  const waterGeo = new THREE.CylinderGeometry(0.021, 0.021, 0.072, 24);
  const waterMat = new THREE.MeshStandardMaterial({
    color: 0x0ea5e9,
    emissive: 0x0284c7,
    emissiveIntensity: 0.35,
    transparent: true,
    opacity: 0.85,
    roughness: 0.2,
  });
  const waterMesh = new THREE.Mesh(waterGeo, waterMat);
  waterMesh.position.y = -0.018;
  zAligned.add(waterMesh);

  // 3. Bottle label band
  const labelGeo = new THREE.CylinderGeometry(0.0236, 0.0236, 0.026, 28);
  const labelMat = new THREE.MeshStandardMaterial({
    color: 0xf8fafc,
    roughness: 0.35,
    metalness: 0.1,
  });
  const labelMesh = new THREE.Mesh(labelGeo, labelMat);
  labelMesh.position.y = -0.008;
  zAligned.add(labelMesh);

  // 4. Tapered bottle neck
  const neckGeo = new THREE.CylinderGeometry(0.0095, 0.0225, 0.026, 24);
  const neckMesh = new THREE.Mesh(neckGeo, petMat);
  neckMesh.position.y = 0.044;
  zAligned.add(neckMesh);

  // 5. Royal blue bottle cap
  const capGeo = new THREE.CylinderGeometry(0.0108, 0.0108, 0.012, 24);
  const capMat = new THREE.MeshStandardMaterial({
    color: 0x2563eb,
    emissive: 0x1d4ed8,
    emissiveIntensity: 0.3,
    roughness: 0.25,
  });
  const capMesh = new THREE.Mesh(capGeo, capMat);
  capMesh.position.y = 0.062;
  zAligned.add(capMesh);

  bottle.add(zAligned);
  return bottle;
}

/**
 * Create a workspace station pedestal + glowing coaster pad from Z = 0 up to targetZ.
 */
function createStationPedestal(x: number, y: number, bottleBottomZ: number, accentHex: number): THREE.Group {
  const group = new THREE.Group();
  const h = Math.max(0.015, bottleBottomZ);

  // Vertical support column
  const colGeo = new THREE.CylinderGeometry(0.014, 0.018, h, 20);
  colGeo.rotateX(Math.PI / 2);
  const colMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b,
    metalness: 0.6,
    roughness: 0.4,
    transparent: true,
    opacity: 0.72,
  });
  const colMesh = new THREE.Mesh(colGeo, colMat);
  colMesh.position.set(x, y, h * 0.5);
  group.add(colMesh);

  // Glowing top coaster pad
  const padGeo = new THREE.CylinderGeometry(0.042, 0.042, 0.006, 32);
  padGeo.rotateX(Math.PI / 2);
  const padMat = new THREE.MeshStandardMaterial({
    color: accentHex,
    emissive: accentHex,
    emissiveIntensity: 0.45,
    metalness: 0.4,
    roughness: 0.3,
  });
  const padMesh = new THREE.Mesh(padGeo, padMat);
  padMesh.position.set(x, y, h);
  group.add(padMesh);

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
  const isCarbon = (design.links.tube_profile || "").startsWith("cf_");
  const isWideTube = (design.links.tube_profile || "").includes("30x2");
  const linkRadius = isWideTube ? 0.022 : 0.019;

  const silverShellMat = isGhost
    ? new THREE.MeshStandardMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.16,
        roughness: 0.4,
        depthWrite: false,
      })
    : new THREE.MeshStandardMaterial({
        color: isCarbon ? 0x334155 : 0xdbe4ee,
        metalness: isCarbon ? 0.45 : 0.72,
        roughness: isCarbon ? 0.38 : 0.26,
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

  // 0. Base Controller Cabinet & Pedestal
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

  // 1. J1 Base Yaw Turret at (0, 0, 0), rotates around +Z
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

  // 2. J2 Shoulder Pitch at (0, 0, baseH), rotates around -Y
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

  // 3. J3 Elbow Pitch at (L1, 0, 0), rotates around -Y
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

  // 4. J4 Wrist Pitch at (L2, 0, 0), rotates around -Y
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
    // ISO Tool Flange & Parallel 2-Finger Gripper Body at X = L3
    const flangeGeo = new THREE.CylinderGeometry(0.022, 0.022, 0.012, 24);
    flangeGeo.rotateZ(-Math.PI / 2);
    const flangeMesh = new THREE.Mesh(flangeGeo, darkSlateMat);
    flangeMesh.position.set(L3 - 0.012, 0, 0);
    wristGripperGroup.add(flangeMesh);

    const gripperRailGeo = new THREE.BoxGeometry(0.016, 0.064, 0.022);
    const gripperRail = new THREE.Mesh(gripperRailGeo, darkSlateMat);
    gripperRail.position.set(L3 - 0.002, 0, 0);
    wristGripperGroup.add(gripperRail);

    const fingerGeo = new THREE.BoxGeometry(0.038, 0.006, 0.024);
    const fingerMat = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      metalness: 0.7,
      roughness: 0.25,
    });
    leftFinger = new THREE.Mesh(fingerGeo, fingerMat);
    leftFinger.position.set(L3 + 0.018, 0.022, 0);
    wristGripperGroup.add(leftFinger);

    rightFinger = new THREE.Mesh(fingerGeo, fingerMat);
    rightFinger.position.set(L3 + 0.018, -0.022, 0);
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
    this.scene.background = new THREE.Color(0x0b101b);

    const w = Math.max(320, container.clientWidth || 480);
    const h = Math.max(260, container.clientHeight || 360);

    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.02, 12.0);
    this.camera.position.set(0.76, 0.56, 0.72);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;

    container.innerHTML = "";
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0.22, 0.24, 0.0);
    this.controls.update();

    const ambient = new THREE.AmbientLight(0xdbeafe, 0.85);
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

    const grid = new THREE.GridHelper(1.6, 16, 0x334155, 0x1e293b);
    this.scene.add(grid);

    this.worldZUp = new THREE.Group();
    this.worldZUp.rotation.x = -Math.PI / 2;
    this.scene.add(this.worldZUp);

    const sphereGeo = new THREE.SphereGeometry(0.009, 16, 16);
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

    if (trace && trace.ee_des && trace.ee_des.length > 1) {
      const targetPts = trace.ee_des.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      const targetGeo = new THREE.BufferGeometry().setFromPoints(targetPts);
      const targetMat = new THREE.LineBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.78,
      });
      this.targetLine = new THREE.Line(targetGeo, targetMat);
      this.worldZUp.add(this.targetLine);

      const actPts = trace.ee_act.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      const actGeo = new THREE.BufferGeometry().setFromPoints(actPts);
      const actMat = new THREE.LineBasicMaterial({
        color: isPassedDesign ? 0x10b981 : 0xef4444,
        transparent: true,
        opacity: 0.92,
      });
      this.actualLine = new THREE.Line(actGeo, actMat);
      this.worldZUp.add(this.actualLine);

      // Error tether line between ee_des(t) and ee_act(t)
      const tetherGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(),
        new THREE.Vector3(),
      ]);
      const tetherMat = new THREE.LineDashedMaterial({
        color: isPassedDesign ? 0x38bdf8 : 0xef4444,
        dashSize: 0.015,
        gapSize: 0.01,
        transparent: true,
        opacity: 0.85,
      });
      this.errorTetherLine = new THREE.Line(tetherGeo, tetherMat);
      this.worldZUp.add(this.errorTetherLine);

      // Build Pickup Station A (at t ~ 1.15s) and Target Station B (at t ~ 4.85s)
      const graspIdx = Math.min(trace.ee_des.length - 1, Math.floor(trace.ee_des.length * 0.19));
      const releaseIdx = Math.min(trace.ee_des.length - 1, Math.floor(trace.ee_des.length * 0.81));
      const pPickup = trace.ee_des[graspIdx];
      const pTarget = trace.ee_des[releaseIdx];

      this.stationsGroup = new THREE.Group();
      // Bottle bottom is ~0.088 m below the grasp origin
      const bottleBottomOffset = 0.088;
      this.stationsGroup.add(
        createStationPedestal(pPickup[0], pPickup[1], pPickup[2] - bottleBottomOffset, 0x38bdf8)
      );
      this.stationsGroup.add(
        createStationPedestal(pTarget[0], pTarget[1], pTarget[2] - bottleBottomOffset, 0x10b981)
      );
      this.worldZUp.add(this.stationsGroup);
    }
  }

  /**
   * Control visibility of progressive assembly stages (1..7) during the "Assembling Robot Arm" animation.
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
   * Update robot pose, gripper fingers, 3D water bottle position, and joint torque-arrow status at frame `frameIdx`.
   */
  public updateFrame(trace: TraceData, frameIdx: number): void {
    if (!this.actualRig || !trace || !trace.t.length) return;
    const idx = Math.max(0, Math.min(trace.t.length - 1, frameIdx));

    const qAct = trace.q_act[idx] || [0, 0, 0, 0];
    const qDes = trace.q_des[idx] || [0, 0, 0, 0];
    const tauCmd = trace.tau_cmd[idx] || [0, 0, 0, 0];
    const tauAvail = trace.tau_avail[idx] || [1, 1, 1, 1];
    const eeAct = trace.ee_act[idx] || [0, 0, 0];
    const eeDes = trace.ee_des[idx] || [0, 0, 0];

    this.actualRig.joints.J1.pivot.rotation.z = qAct[0];
    this.actualRig.joints.J2.pivot.rotation.y = -qAct[1];
    this.actualRig.joints.J3.pivot.rotation.y = -qAct[2];
    this.actualRig.joints.J4.pivot.rotation.y = -qAct[3];

    if (this.ghostRig) {
      this.ghostRig.joints.J1.pivot.rotation.z = qDes[0];
      this.ghostRig.joints.J2.pivot.rotation.y = -qDes[1];
      this.ghostRig.joints.J3.pivot.rotation.y = -qDes[2];
      this.ghostRig.joints.J4.pivot.rotation.y = -qDes[3];
    }

    this.eeActualMarker.position.set(eeAct[0], eeAct[1], eeAct[2]);
    this.eeTargetMarker.position.set(eeDes[0], eeDes[1], eeDes[2]);

    // Update error tether line between desired and actual EE
    if (this.errorTetherLine) {
      const posAttr = this.errorTetherLine.geometry.getAttribute("position") as THREE.BufferAttribute;
      if (posAttr) {
        posAttr.setXYZ(0, eeDes[0], eeDes[1], eeDes[2]);
        posAttr.setXYZ(1, eeAct[0], eeAct[1], eeAct[2]);
        posAttr.needsUpdate = true;
        this.errorTetherLine.computeLineDistances();
      }
    }

    // Water Bottle Pick-and-Transfer state machine
    const graspIdx = Math.min(trace.t.length - 1, Math.floor(trace.t.length * 0.19));
    const releaseIdx = Math.min(trace.t.length - 1, Math.floor(trace.t.length * 0.81));
    const pPickup = trace.ee_des[graspIdx] || eeDes;
    const pTarget = trace.ee_des[releaseIdx] || eeDes;
    const isGrasped = idx >= graspIdx && idx <= releaseIdx;

    if (this.actualRig.leftFinger && this.actualRig.rightFinger) {
      const spread = isGrasped ? 0.0155 : 0.025;
      this.actualRig.leftFinger.position.y = spread;
      this.actualRig.rightFinger.position.y = -spread;
    }

    if (this.waterBottle && this.ghostBottle) {
      if (idx < graspIdx) {
        // Bottle waiting on Pickup Station A
        this.waterBottle.position.set(pPickup[0], pPickup[1], pPickup[2]);
        this.ghostBottle.position.set(pPickup[0], pPickup[1], pPickup[2]);
      } else if (idx <= releaseIdx) {
        // Bottle carried by the gripper!
        // Actual bottle follows ee_act (droops if J2 saturates!); ghost bottle follows ee_des
        this.waterBottle.position.set(eeAct[0], eeAct[1], eeAct[2]);
        this.ghostBottle.position.set(eeDes[0], eeDes[1], eeDes[2]);
      } else {
        // Released at end of transfer
        const endActual = this.isPassedDesign ? pTarget : (trace.ee_act[releaseIdx] || eeAct);
        this.waterBottle.position.set(endActual[0], endActual[1], endActual[2]);
        this.ghostBottle.position.set(pTarget[0], pTarget[1], pTarget[2]);
      }
    }

    // Color-code each joint's curved rotation arrows & accent ring by torque load ratio
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
