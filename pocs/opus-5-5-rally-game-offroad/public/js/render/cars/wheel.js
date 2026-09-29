import * as THREE from 'three';
import { createBuilder, place } from './builder.js';

function tireGeometry(R, w, rr) {
  const pts = [
    [rr + 0.01, -w * 0.4],
    [R - 0.09, -w * 0.5],
    [R - 0.035, -w * 0.47],
    [R - 0.008, -w * 0.4],
    [R, -w * 0.3],
    [R, w * 0.3],
    [R - 0.008, w * 0.4],
    [R - 0.035, w * 0.47],
    [R - 0.09, w * 0.5],
    [rr + 0.01, w * 0.4],
  ].map(([r, y]) => new THREE.Vector2(r, y));
  const g = new THREE.LatheGeometry(pts, 40);
  g.rotateZ(Math.PI / 2);
  return g;
}

function tread(b, R, w) {
  const n = 30;
  for (let k = 0; k < n; k++) {
    for (const row of [-1, 1]) {
      const a = ((k + (row > 0 ? 0.5 : 0)) / n) * Math.PI * 2;
      const len = ((2 * Math.PI * R) / n) * 0.55;
      b.add(place(new THREE.BoxGeometry(w * 0.34, 0.034, len), row * w * 0.19, Math.cos(a) * (R + 0.006), Math.sin(a) * (R + 0.006), a), 'rubber');
      b.add(place(new THREE.BoxGeometry(w * 0.16, 0.03, len * 0.8), row * w * 0.43, Math.cos(a + 0.05) * (R - 0.018), Math.sin(a + 0.05) * (R - 0.018), a + 0.05), 'rubber');
    }
  }
}

function rimFace(b, s, rr, w, style) {
  const x = s * w * 0.33;
  b.cyl(rr, rr, 0.03, 'rim', x, 0, 0, 0, 0, Math.PI / 2, 28);
  b.torus(rr * 0.96, 0.028, 'rimLip', x + s * 0.015, 0, 0, 0, Math.PI / 2, 0);
  const bolts = style === 'beadlock' ? 18 : 0;
  for (let k = 0; k < bolts; k++) {
    const a = (k / bolts) * Math.PI * 2;
    b.cyl(0.011, 0.011, 0.03, 'chrome', x + s * 0.03, Math.cos(a) * rr * 0.9, Math.sin(a) * rr * 0.9, 0, 0, Math.PI / 2, 6);
  }
  const spokes = style === 'steel' ? 0 : 6;
  for (let k = 0; k < spokes; k++) {
    const a = (k / spokes) * Math.PI * 2;
    b.add(place(new THREE.BoxGeometry(0.03, rr * 0.62, 0.075), x + s * 0.02, Math.cos(a) * rr * 0.5, Math.sin(a) * rr * 0.5, a), 'rimLip');
  }
  if (style === 'steel') {
    for (let k = 0; k < 8; k++) {
      const a = (k / 8) * Math.PI * 2;
      b.cyl(0.035, 0.035, 0.035, 'trim', x + s * 0.012, Math.cos(a) * rr * 0.62, Math.sin(a) * rr * 0.62, 0, 0, Math.PI / 2, 10);
    }
  }
  b.cyl(rr * 0.24, rr * 0.28, 0.06, 'rimLip', x + s * 0.03, 0, 0, 0, 0, Math.PI / 2, 16);
  for (let k = 0; k < 6; k++) {
    const a = (k / 6) * Math.PI * 2;
    b.cyl(0.012, 0.012, 0.05, 'chrome', x + s * 0.055, Math.cos(a) * rr * 0.16, Math.sin(a) * rr * 0.16, 0, 0, Math.PI / 2, 6);
  }
  if (style === 'ctis') {
    b.cyl(rr * 0.2, rr * 0.3, 0.14, 'rimLip', x + s * 0.1, 0, 0, 0, 0, Math.PI / 2, 14);
    b.cyl(0.03, 0.03, 0.12, 'chrome', x + s * 0.2, 0, 0, 0, 0, Math.PI / 2, 8);
  }
}

export function buildWheel(materials, spec, style) {
  const R = spec.wheelR;
  const w = R * 0.78;
  const rr = R * 0.55;
  const group = new THREE.Group();
  const spin = new THREE.Group();
  group.add(spin);
  const b = createBuilder();
  b.add(tireGeometry(R, w, rr), 'rubber');
  tread(b, R, w);
  b.cyl(rr + 0.012, rr + 0.012, w * 0.78, 'trim', 0, 0, 0, 0, 0, Math.PI / 2, 24);
  b.both((s) => rimFace(b, s, rr, w, style));
  spin.add(b.build(materials));
  return { group, spin, triangles: b.triangles() };
}
