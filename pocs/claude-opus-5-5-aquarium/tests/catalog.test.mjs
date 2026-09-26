import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TANK, RACK_COLORS, MATERIALS, FISH, DECOR, MAX_GRASS } from '../web/catalog.mjs';
import { grassSpots, CLUMPS_PER_LEVEL, AIR_STONE } from '../web/layout.mjs';
import { PATTERNS } from '../web/patterns.mjs';

const unique = list => new Set(list.map(i => i.id)).size === list.length;

test('the rack offers between 6 and 10 colors, each a real hex except Natural', () => {
  assert.ok(RACK_COLORS.length >= 6 && RACK_COLORS.length <= 10);
  assert.ok(unique(RACK_COLORS));
  for (const c of RACK_COLORS) {
    if (c.id === 'natural') assert.equal(c.hex, null);
    else assert.match(c.hex, /^#[0-9a-f]{6}$/);
  }
});

test('the rack offers up to 10 materials, including wood, steel and bricks', () => {
  assert.ok(MATERIALS.length <= 10 && MATERIALS.length >= 3);
  assert.ok(unique(MATERIALS));
  for (const id of ['wood', 'steel', 'bricks']) assert.ok(MATERIALS.some(m => m.id === id), id);
  for (const m of MATERIALS) {
    assert.ok(PATTERNS.includes(m.pattern), `${m.id} needs a texture pattern`);
    assert.ok(m.metalness >= 0 && m.metalness <= 1);
    assert.ok(m.roughness >= 0 && m.roughness <= 1);
  }
});

test('up to 10 fish species, each small enough to turn inside the tank', () => {
  assert.ok(FISH.length <= 10 && FISH.length >= 1);
  assert.ok(unique(FISH));
  for (const f of FISH) {
    assert.ok(f.len * 2 < TANK.depth, `${f.id} is too long for the tank depth`);
    assert.ok(f.band[0] >= 0 && f.band[1] <= 1 && f.band[0] < f.band[1]);
    assert.ok(f.speed > 0);
  }
});

test('up to 10 decorations counting seagrass, including sunken car, ship and plane', () => {
  assert.ok(DECOR.length + 1 <= 10);
  assert.ok(unique(DECOR));
  for (const id of ['car', 'ship', 'plane']) assert.ok(DECOR.some(d => d.id === id), id);
  for (const d of DECOR) assert.ok(d.spots.length > 0, `${d.id} needs a spot on the sand`);
});

test('decorations stay inside the glass and never sit on top of each other', () => {
  const spots = DECOR.flatMap(d => d.spots.map(([x, z, r]) => ({ id: d.id, x, z, r })));
  for (const s of spots) {
    assert.ok(Math.abs(s.x) + s.r <= TANK.width / 2, `${s.id} pokes through a side pane`);
    assert.ok(Math.abs(s.z) + s.r <= TANK.depth / 2, `${s.id} pokes through the front or back pane`);
  }
  for (let i = 0; i < spots.length; i++) {
    for (let j = i + 1; j < spots.length; j++) {
      const a = spots[i];
      const b = spots[j];
      assert.ok(Math.hypot(a.x - b.x, a.z - b.z) >= a.r + b.r, `${a.id} overlaps ${b.id}`);
    }
  }
});

test('every grass level has room to plant, and more grass never grows through decorations', () => {
  const spots = grassSpots();
  assert.equal(spots.length, MAX_GRASS * CLUMPS_PER_LEVEL);
  const blocked = [...DECOR.flatMap(d => d.spots), AIR_STONE];
  for (const g of spots) {
    assert.ok(Math.abs(g.x) < TANK.width / 2 && Math.abs(g.z) < TANK.depth / 2);
    for (const [x, z, r] of blocked) assert.ok(Math.hypot(g.x - x, g.z - z) >= r, 'grass inside a decoration');
  }
});

test('the first grass levels plant the back of the tank so the view stays open', () => {
  const spots = grassSpots();
  const avg = list => list.reduce((a, g) => a + g.z, 0) / list.length;
  assert.ok(avg(spots.slice(0, CLUMPS_PER_LEVEL)) < avg(spots.slice(-CLUMPS_PER_LEVEL)));
});
