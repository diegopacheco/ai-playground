import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TRACKS, project, gridSlots } from '../public/js/core/tracks.js';
import { worldFor } from './geo-fixture.js';

const built = TRACKS.map((def) => ({ def, ...worldFor(def) }));

test('the game ships one track per California location the player asked for', () => {
  assert.deepEqual(TRACKS.map((t) => t.city), ['San Francisco', 'Los Angeles', 'Lake Tahoe', 'Yosemite']);
});

for (const { def, track, terrain, props } of built) {
  test(`${def.city}: no hairpin is tighter than a 4x4 can take, so every corner is driveable`, () => {
    let minRadius = Infinity;
    for (let i = 0; i < track.count; i++) minRadius = Math.min(minRadius, 1 / Math.max(Math.abs(track.curvature[i]), 1e-6));
    assert.ok(minRadius > 40, `tightest radius ${minRadius.toFixed(1)}m`);
  });

  test(`${def.city}: separate parts of the loop never touch, so nobody can shortcut across`, () => {
    const skip = Math.round(90 / track.spacing);
    let gap = Infinity;
    for (let i = 0; i < track.count; i += 2) {
      for (let j = 0; j < track.count; j += 2) {
        let d = Math.abs(i - j);
        d = Math.min(d, track.count - d);
        if (d < skip) continue;
        gap = Math.min(gap, Math.hypot(track.xs[i] - track.xs[j], track.zs[i] - track.zs[j]));
      }
    }
    assert.ok(gap > def.width * 3, `closest approach ${gap.toFixed(0)}m`);
  });

  test(`${def.city}: the road surface is flat across its width, so cars do not roll over on straight road`, () => {
    let worst = 0;
    for (let i = 0; i < track.count; i += 7) {
      const h = track.heading[i];
      const rx = -Math.cos(h);
      const rz = Math.sin(h);
      const w = track.halfWidth * 0.8;
      const a = terrain.heightAt(track.xs[i] + rx * w, track.zs[i] + rz * w);
      const b = terrain.heightAt(track.xs[i] - rx * w, track.zs[i] - rz * w);
      worst = Math.max(worst, Math.abs(a - b) / (2 * w));
    }
    assert.ok(worst < 0.12, `worst cross slope ${worst.toFixed(3)}`);
  });

  test(`${def.city}: the road stays above the lake so the race never goes underwater`, () => {
    if (def.water === null) return;
    for (let i = 0; i < track.count; i++) assert.ok(terrain.roadY[i] > terrain.water + 1);
  });

  test(`${def.city}: trees and rocks are never planted on the racing line`, () => {
    for (const o of props.obstacles) {
      const p = project(track, o.x, o.z);
      assert.ok(p.dist > track.halfWidth + 2, `obstacle ${p.dist.toFixed(1)}m from centerline`);
    }
    assert.ok(props.trees.length > def.trees.count * 0.5, 'forest is dense enough to look real');
  });

  test(`${def.city}: the 4 starting slots do not overlap and sit just past the start line`, () => {
    const slots = gridSlots(track, 4);
    for (let a = 0; a < 4; a++) {
      assert.ok(slots[a].i > 0 && slots[a].i < track.count * 0.1);
      for (let b = a + 1; b < 4; b++) assert.ok(Math.hypot(slots[a].x - slots[b].x, slots[a].z - slots[b].z) > 4);
    }
  });
}

test('projection gives a signed lateral offset so the game knows which side of the road a car is on', () => {
  const { track } = built[0];
  const i = 300;
  const h = track.heading[i];
  const right = project(track, track.xs[i] - Math.cos(h) * 3, track.zs[i] + Math.sin(h) * 3, i);
  const left = project(track, track.xs[i] + Math.cos(h) * 3, track.zs[i] - Math.sin(h) * 3, i);
  assert.ok(Math.abs(right.lateral - 3) < 0.2);
  assert.ok(Math.abs(left.lateral + 3) < 0.2);
});
