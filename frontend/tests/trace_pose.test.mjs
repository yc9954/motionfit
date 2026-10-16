import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import ts from 'typescript';
import * as THREE from 'three';

const frontend = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const scratch = mkdtempSync(join(frontend, '.trace-contract-'));
function compile(name) {
  const source = readFileSync(join(frontend, 'src', `${name}.ts`), 'utf8');
  const result = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } });
  const path = join(scratch, `${name}.mjs`);
  writeFileSync(path, result.outputText);
  return import(pathToFileURL(path).href);
}
let viewer, pose;
try { [viewer, pose] = await Promise.all([compile('viewer'), compile('pose')]); }
finally { rmSync(scratch, { recursive: true, force: true }); }
const makeTrace = () => ({
  fps: 100, t: [0, 0.01],
  q_act: [[0, 0, 0, 0], [0.4, 1.1, -0.6, 0.2]],
  q_des: [[0, 0, 0, 0], [-0.2, 0.9, -0.4, -0.7]],
  qd_act: [], qd_des: [],
  ee_act: [[0.3, 0, 0.2], [0.321, -0.123, 0.456]],
  ee_des: [[0.3, 0, 0.2], [0.654, 0.234, 0.567]],
  tau_req: [[0, 0, 0, 0], [20, 8, 5, 1]],
  tau_cmd: [[0, 0, 0, 0], [10, 8, 5, 1]],
  tau_avail: [[10, 10, 10, 10], [10, 10, 10, 10]],
});
const design = {
  design_id: 'test', topology: 'arm4', payload_kg: 0.5,
  links: { base_height_m: 0.12, upper_arm_m: 0.3, forearm_m: 0.25, wrist_to_tool_m: 0.08, tube_profile: 'al_tube_20x1.5' },
  joints: Object.fromEntries(['J1', 'J2', 'J3', 'J4'].map(key => [key, { actuator_id: 'test-actuator', gearbox_id: null }])),
};
function sceneOnlyViewport() {
  const viewport = Object.create(viewer.RobotArmViewport.prototype);
  Object.assign(viewport, {
    disposed: false, actualRig: null, ghostRig: null, targetLine: null, actualLine: null, errorTetherLine: null,
    worldZUp: new THREE.Group(), camera: new THREE.PerspectiveCamera(40, 1, 0.01, 30),
    controls: { target: new THREE.Vector3(), update() {} },
    showAxisOverlays: true, showGhostArm: true,
    eeActualMarker: new THREE.Mesh(), eeTargetMarker: new THREE.Mesh(),
  });
  return viewport;
}

test('3D rig angles, EE markers and paths use the supplied simulation records verbatim', () => {
  const trace = makeTrace();
  const viewport = sceneOnlyViewport();
  viewport.setDesignAndTrace(design, trace, false, { parts: [{ part_id: 'test-actuator', dimensions_mm: [28.5, 46.5, 34] }] });
  viewport.updateFrame(trace, 1);
  const actual = viewport.actualRig.joints, ghost = viewport.ghostRig.joints;
  assert.equal(actual.J1.pivot.rotation.z, trace.q_act[1][0]);
  for (let j = 1; j < 4; j++) {
    assert.equal(actual[`J${j + 1}`].pivot.rotation.y, -trace.q_act[1][j]);
    assert.equal(ghost[`J${j + 1}`].pivot.rotation.y, -trace.q_des[1][j]);
  }
  assert.deepEqual(viewport.eeActualMarker.position.toArray(), trace.ee_act[1]);
  assert.deepEqual(viewport.eeTargetMarker.position.toArray(), trace.ee_des[1]);
  const actualPath = viewport.actualLine.geometry.getAttribute('position');
  assert.ok(Math.abs(actualPath.getX(1) - trace.ee_act[1][0]) < 1e-6);
  assert.equal(actual.J1.effortMaterial.color.getHex(), 0xef4444, 'overload uses requested torque, not saturated command');
  const before = actual.J2.pivot.rotation.y;
  viewport.setDesignAndTrace(design, trace, true);
  viewport.updateFrame(trace, 1);
  assert.equal(viewport.actualRig.joints.J2.pivot.rotation.y, before, 'pass/fail status cannot invent a different motion');
});

test('unavailable or malformed trace data is rejected without generating another trajectory', () => {
  const trace = makeTrace();
  trace.q_act[1] = [0, Number.NaN, 0, 0];
  assert.throws(() => viewer.resolveTraceFrame(trace, 1), /Invalid simulation trace/);
  assert.throws(() => viewer.resolveTraceFrame({ t: [] }, 0), /no frames/);
});

test('initial geometry contains no target path or implied simulation result', () => {
  const viewport = sceneOnlyViewport();
  viewport.setDesignAndTrace(design, null, false);
  assert.ok(viewport.actualRig);
  assert.equal(viewport.actualLine, null);
  assert.equal(viewport.targetLine, null);
  assert.equal(viewport.eeActualMarker.visible, false);
  viewport.setGhostArmVisible(true);
  assert.equal(viewport.ghostRig.root.visible, false);
  assert.equal(viewport.eeTargetMarker.visible, false);
});

test('actuator envelopes and tube diameters use catalog millimetres without family guesses', () => {
  assert.deepEqual(viewer.partEnvelope({ part_id: 'any-name', dimensions_mm: [28.5, 46.5, 34] }), [0.0285, 0.0465, 0.034]);
  assert.equal(viewer.partEnvelope({ part_id: 'AK80-9' }), null);
  assert.equal(viewer.partEnvelope({ part_id: 'bad', dimensions_mm: [28, 0, 12] }), null);
  assert.equal(viewer.tubeRadius('al_tube_20x1.5'), 0.01);
  assert.equal(viewer.tubeRadius('cf_tube_25x1.5'), 0.0125);
});

test('video overlay draws measured coordinates only when their confidence is sufficient', () => {
  const calls = [];
  const ctx = Object.fromEntries(['save', 'restore', 'beginPath', 'stroke', 'fill'].map(name => [name, () => {}]));
  for (const name of ['moveTo', 'lineTo', 'arc']) ctx[name] = (...args) => calls.push([name, ...args]);
  const points = Array.from({ length: 33 }, () => ({ x: 0.1, y: 0.2, z: 0, visibility: 0 }));
  pose.drawPoseOverlay(ctx, 100, 200, points);
  assert.equal(calls.length, 0, 'no invented skeleton for missing measurements');
  for (const i of [12, 14, 16, 20, 24]) points[i] = { x: i / 100, y: 0.25, z: 0, visibility: 0.9 };
  pose.drawPoseOverlay(ctx, 100, 200, points);
  assert.deepEqual(calls.find(call => call[0] === 'moveTo'), ['moveTo', 24, 50]);
  assert.equal(calls.filter(call => call[0] === 'arc').length, 5);
});

test('pose frames preserve raw metre coordinates including hip and reject missing measurements', () => {
  assert.equal(pose.measuredPoseFrame(undefined, undefined, 0), null);
  const world = Array.from({ length: 33 }, (_, i) => ({ x: i / 100, y: -i / 100, z: i / 200, visibility: 0.9 }));
  const normalized = world.map(p => ({ visibility: p.visibility }));
  const frame = pose.measuredPoseFrame(world, normalized, 1.25, 'right');
  assert.deepEqual(frame.shoulder, [0.12, -0.12, 0.06]);
  assert.deepEqual(frame.hip, [0.24, -0.24, 0.12]);
  assert.equal(frame.t, 1.25);
  normalized[16].visibility = 0.1;
  assert.equal(pose.measuredPoseFrame(world, normalized, 1.25), null);
  assert.throws(() => pose.validatePoseTracking(100, 79), /No substitute motion was generated/);
  assert.throws(() => pose.validatePoseTracking(20, 14), /Pose tracking failed/);
  assert.ok(Math.abs(pose.validatePoseTracking(100, 80) - 0.2) < 1e-9);
});

test('viewport disposal releases graphics, observer and canvas resources exactly once', () => {
  const viewport = sceneOnlyViewport();
  const counts = {};
  const count = name => () => { counts[name] = (counts[name] ?? 0) + 1; };
  const geometry = new THREE.BoxGeometry(), material = new THREE.MeshBasicMaterial();
  geometry.dispose = count('geometry'); material.dispose = count('material');
  viewport.scene = new THREE.Scene(); viewport.scene.add(new THREE.Mesh(geometry, material));
  viewport.resizeObserver = { disconnect: count('observer') };
  viewport.controls.dispose = count('controls');
  viewport.renderer = { renderLists: { dispose: count('renderLists') }, dispose: count('renderer'), forceContextLoss: count('context'), domElement: { remove: count('canvas') } };
  viewport.dispose(); viewport.dispose();
  assert.deepEqual(counts, { observer: 1, controls: 1, material: 1, geometry: 1, renderLists: 1, renderer: 1, context: 1, canvas: 1 });
});
