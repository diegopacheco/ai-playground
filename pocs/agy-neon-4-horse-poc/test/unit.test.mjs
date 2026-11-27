import test from 'node:test';
import assert from 'node:assert/strict';
import { evaluateGaitPose } from '../public/js/gait.js';
import {
  getMountainRidgeHeight,
  getTerrainHeight,
  getTerrainNormal,
  getTrailCenterX,
  getTrailMask
} from '../public/js/terrain-math.js';

test('gallop reaches an airborne suspension phase while walk keeps constant ground support', () => {
  let gallopSuspended = false;
  let walkSuspended = false;

  for (let i = 0; i < 100; i += 1) {
    const cycle = i / 100;
    if (evaluateGaitPose(cycle, 'gallop').suspended) gallopSuspended = true;
    if (evaluateGaitPose(cycle, 'walk').suspended) walkSuspended = true;
  }

  assert.equal(gallopSuspended, true);
  assert.equal(walkSuspended, false);
});

test('foreleg knee bends backward while hindleg hock bends forward during swing', () => {
  let maxForeKneeFlex = -Infinity;
  let minHindHockFlex = Infinity;

  for (let i = 0; i < 120; i += 1) {
    const pose = evaluateGaitPose(i / 120, 'gallop');
    const hindHock = pose.legs[0].lower;
    const foreKnee = pose.legs[2].lower;
    if (foreKnee > maxForeKneeFlex) maxForeKneeFlex = foreKnee;
    if (hindHock < minHindHockFlex) minHindHockFlex = hindHock;
  }

  assert.ok(maxForeKneeFlex > 1.0);
  assert.ok(minHindHockFlex < -0.8);
});

test('gait pose wraps cleanly from cycle 0 to cycle 1 without discontinuity', () => {
  const start = evaluateGaitPose(0, 'gallop');
  const end = evaluateGaitPose(1, 'gallop');

  assert.ok(Math.abs(start.bodyY - end.bodyY) < 1e-6);
  assert.ok(Math.abs(start.bodyPitch - end.bodyPitch) < 1e-6);
  for (let i = 0; i < 4; i += 1) {
    assert.ok(Math.abs(start.legs[i].upper - end.legs[i].upper) < 1e-6);
    assert.ok(Math.abs(start.legs[i].lower - end.legs[i].lower) < 1e-6);
  }
});

test('colorado trail mask centers on trail path and foothills rise outward with valid normals', () => {
  const z = 42.5;
  const cx = getTrailCenterX(z);

  assert.equal(getTrailMask(cx, z), 1);
  assert.equal(getTrailMask(cx + 12, z), 0);

  const valleyY = getTerrainHeight(cx, z);
  const foothillY = getTerrainHeight(cx + 55, z);
  assert.ok(foothillY > valleyY + 10);

  const normal = getTerrainNormal(cx, z);
  const mag = Math.hypot(normal.x, normal.y, normal.z);
  assert.ok(Math.abs(mag - 1) < 1e-6);
  assert.ok(normal.y > 0.7);
});

test('rocky mountain amphitheater rises to alpine peak elevations and tapers at boundaries', () => {
  const innerEdge = getMountainRidgeHeight(1.36, 0);
  const peakCenter = getMountainRidgeHeight(1.36, 0.5);
  const outerEdge = getMountainRidgeHeight(1.36, 1);

  assert.ok(Math.abs(innerEdge) < 1e-6);
  assert.ok(Math.abs(outerEdge) < 1e-6);
  assert.ok(peakCenter > 70);
});
