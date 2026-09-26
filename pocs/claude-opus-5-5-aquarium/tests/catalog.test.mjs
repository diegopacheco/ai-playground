import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TANK, RACK_COLORS, MATERIALS, FISH, DECOR, STONES, BUSHES, MAX_GRASS } from '../web/catalog.mjs';
import { grassSpots, carpetSpots, covered, CLUMPS_PER_LEVEL, CARPET_STEP, AIR_STONE } from '../web/layout.mjs';
import { PATTERNS, pathMask } from '../web/patterns.mjs';

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

test('every grass level has room to plant, and grass never grows through decorations or dragon stones', () => {
  const spots = grassSpots();
  assert.equal(spots.length, MAX_GRASS * CLUMPS_PER_LEVEL);
  const blocked = [...DECOR.flatMap(d => d.spots), ...STONES, AIR_STONE];
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

test('dragon stones stay inside the glass and never crash into a decoration or the air stone', () => {
  const others = [...DECOR.flatMap(d => d.spots), AIR_STONE];
  STONES.forEach(([x, z, r, h], i) => {
    assert.ok(Math.abs(x) + r <= TANK.width / 2 && Math.abs(z) + r <= TANK.depth / 2, `stone ${i} pokes through the glass`);
    assert.ok(h < TANK.water - 0.1, `stone ${i} breaks the water surface`);
    for (const [ox, oz, or] of others) assert.ok(Math.hypot(x - ox, z - oz) >= r + or, `stone ${i} overlaps a decoration`);
  });
});

test('bushes grow along the back glass and leave the air stone bubbles free', () => {
  for (const b of BUSHES) {
    assert.ok(Math.abs(b.x) + b.rx <= TANK.width / 2 && Math.abs(b.z) + b.rz <= TANK.depth / 2, `bush at ${b.x} pokes through the glass`);
    assert.ok(b.z < -TANK.depth / 4, 'bushes belong in the background so they never hide the fish');
    assert.ok(Math.abs(b.x - AIR_STONE[0]) >= b.rx + AIR_STONE[2] || Math.abs(b.z - AIR_STONE[1]) >= b.rz + AIR_STONE[2], 'a bush swallows the bubble stream');
    assert.ok(b.h < TANK.water - 0.1);
  }
});

test('the carpet covers the whole floor densely but never grows over the air stone', () => {
  const spots = carpetSpots();
  const area = TANK.width * TANK.depth;
  assert.ok(spots.length >= (area / (CARPET_STEP * CARPET_STEP)) * 0.8, 'the carpet has bald patches');
  for (const s of spots) {
    assert.ok(Math.abs(s.x) < TANK.width / 2 && Math.abs(s.z) < TANK.depth / 2);
    assert.ok(!covered(s.x, s.z, [AIR_STONE]));
  }
});

test('the sand path winds from the back glass to the front glass and stays a narrow path', () => {
  const rows = 40;
  for (let i = 0; i <= rows; i++) {
    const z = (i / rows - 0.5) * (TANK.depth - 0.02);
    let open = 0;
    const cols = 240;
    for (let j = 0; j <= cols; j++) if (pathMask((j / cols - 0.5) * TANK.width, z) > 0.5) open++;
    assert.ok(open > 0, `the path breaks at z=${z.toFixed(2)}`);
    assert.ok(open / cols < 0.15, `the path floods the floor at z=${z.toFixed(2)}`);
  }
});
