import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

const euler = new THREE.Euler();
const quat = new THREE.Quaternion();
const mat4 = new THREE.Matrix4();
const one = new THREE.Vector3(1, 1, 1);
const pos = new THREE.Vector3();

export function place(geo, x = 0, y = 0, z = 0, rx = 0, ry = 0, rz = 0) {
  euler.set(rx, ry, rz);
  quat.setFromEuler(euler);
  pos.set(x, y, z);
  return geo.applyMatrix4(mat4.compose(pos, quat, one));
}

function clean(geo) {
  const g = geo.index ? geo.toNonIndexed() : geo;
  if (!g.attributes.normal) g.computeVertexNormals();
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', g.attributes.position);
  out.setAttribute('normal', g.attributes.normal);
  out.setAttribute('uv', g.attributes.uv || new THREE.BufferAttribute(new Float32Array(g.attributes.position.count * 2), 2));
  return out;
}

function arcPoints(a, sill, steps = 18) {
  if (a.square) {
    return [[a.z - a.r, sill], [a.z - a.r * 0.8, a.y + a.r * 0.9], [a.z + a.r * 0.8, a.y + a.r * 0.9], [a.z + a.r, sill]];
  }
  const t0 = Math.asin(Math.max(-1, Math.min(1, (sill - a.y) / a.r)));
  const out = [];
  for (let k = 0; k <= steps; k++) {
    const ang = Math.PI - t0 + ((2 * t0 - Math.PI) * k) / steps;
    out.push([a.z + Math.cos(ang) * a.r, a.y + Math.sin(ang) * a.r]);
  }
  return out;
}

export function outline(top, sill, arches = []) {
  const under = [];
  for (const a of [...arches].sort((p, q) => p.z - q.z)) under.push(...arcPoints(a, sill));
  const pts = [...top, ...under];
  return pts.filter((p, i) => {
    const q = pts[(i + pts.length - 1) % pts.length];
    return Math.hypot(p[0] - q[0], p[1] - q[1]) > 1e-4;
  });
}

function shapeOf(pts) {
  return new THREE.Shape(pts.map(([z, y]) => new THREE.Vector2(z, y)));
}

export function createBuilder() {
  const bins = new Map();
  const add = (geo, key) => {
    if (!bins.has(key)) bins.set(key, []);
    bins.get(key).push(clean(geo));
    return geo;
  };
  const b = {
    add,
    box: (w, h, d, key, x, y, z, rx, ry, rz) => add(place(new THREE.BoxGeometry(w, h, d), x, y, z, rx, ry, rz), key),
    rbox: (w, h, d, r, key, x, y, z, rx, ry, rz) => add(place(new RoundedBoxGeometry(w, h, d, 2, Math.min(r, w / 2 - 0.001, h / 2 - 0.001, d / 2 - 0.001)), x, y, z, rx, ry, rz), key),
    cyl: (rt, rb, h, key, x, y, z, rx = 0, ry = 0, rz = 0, seg = 18) => add(place(new THREE.CylinderGeometry(rt, rb, h, seg), x, y, z, rx, ry, rz), key),
    torus: (r, t, key, x, y, z, rx = 0, ry = 0, rz = 0, arc = Math.PI * 2) => add(place(new THREE.TorusGeometry(r, t, 6, 28, arc), x, y, z, rx, ry, rz), key),
    tube: (a, c, r, key) => {
      const from = new THREE.Vector3(...a);
      const to = new THREE.Vector3(...c);
      const len = from.distanceTo(to);
      const g = new THREE.CylinderGeometry(r, r, len, 10);
      g.translate(0, len / 2, 0);
      const dir = to.clone().sub(from).normalize();
      g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir));
      g.translate(from.x, from.y, from.z);
      return add(g, key);
    },
    both: (fn) => {
      fn(1);
      fn(-1);
    },
    profile(pts, width, key, bevel = 0.04, x = 0) {
      const depth = Math.max(0.01, width - 2 * bevel);
      const geo = new THREE.ExtrudeGeometry(shapeOf(pts), { depth, bevelEnabled: bevel > 0, bevelThickness: bevel, bevelSize: bevel, bevelOffset: -bevel, bevelSegments: 3, curveSegments: 12, steps: 1 });
      geo.rotateY(-Math.PI / 2);
      geo.translate(depth / 2 + x, 0, 0);
      return add(geo, key);
    },
    side(pts, xAbs, key) {
      b.both((s) => {
        const g = new THREE.ShapeGeometry(shapeOf(pts));
        g.rotateY(-Math.PI / 2);
        g.translate(s * xAbs, 0, 0);
        add(g, key);
      });
    },
    slab(pts, xAbs, thick, key) {
      b.both((s) => {
        const g = new THREE.ExtrudeGeometry(shapeOf(pts), { depth: thick, bevelEnabled: false, curveSegments: 12 });
        g.rotateY(-Math.PI / 2);
        g.translate(s > 0 ? xAbs + thick : -xAbs, 0, 0);
        add(g, key);
      });
    },
    edge(p1, p2, width, key, lift, trim, out, x = 0) {
      const dz = p2[0] - p1[0];
      const dy = p2[1] - p1[1];
      const len = Math.hypot(dz, dy);
      let nz = dy / len;
      let ny = -dz / len;
      if (Math.sign(nz || ny) !== out) {
        nz = -nz;
        ny = -ny;
      }
      const g = new THREE.PlaneGeometry(width, Math.max(0.02, len - 2 * trim));
      place(g, x, (p1[1] + p2[1]) / 2 + ny * lift, (p1[0] + p2[0]) / 2 + nz * lift, Math.atan2(-ny, nz));
      return add(g, key);
    },
    flare(zc, yc, r, yb, outer, xAbs, depth, key) {
      const t0 = Math.asin(Math.max(-1, Math.min(1, (yb - yc) / r)));
      const inner = [];
      for (let k = 0; k <= 16; k++) {
        const ang = t0 + ((Math.PI - 2 * t0) * k) / 16;
        inner.push([zc + Math.cos(ang) * r, yc + Math.sin(ang) * r]);
      }
      const pts = [...outer.map(([dz, dy]) => [zc + dz, yb + dy]), ...inner];
      b.slab(pts, xAbs, depth, key);
    },
    lamp(x, y, z, r, dir = 1, key = 'head', ring = true) {
      b.cyl(r * 1.18, r * 1.18, 0.05, 'chrome', x, y, z, Math.PI / 2, 0, 0, 20);
      b.cyl(r, r, 0.06, key, x, y, z + dir * 0.006, Math.PI / 2, 0, 0, 20);
      if (ring) b.torus(r * 0.72, 0.012, 'drl', x, y, z + dir * 0.038);
    },
    build(materials) {
      const group = new THREE.Group();
      for (const [key, list] of bins) {
        const mesh = new THREE.Mesh(mergeGeometries(list), materials[key]);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);
      }
      return group;
    },
    triangles() {
      let n = 0;
      for (const list of bins.values()) for (const g of list) n += g.attributes.position.count / 3;
      return n;
    },
  };
  return b;
}
