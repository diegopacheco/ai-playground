import * as THREE from 'three';
import { softDot } from './textures.js';
import { SURFACES } from '../core/vehicle.js';
import { ROAD_OFFSET } from '../core/terrain.js';

const BOX = 70;

export function createWeather(scene, weather, quality) {
  if (weather === 'clear') return { update() {} };
  const count = Math.floor((weather === 'rain' ? 9000 : 7000) * quality.particles);
  const pos = new Float32Array(count * (weather === 'rain' ? 6 : 3));
  const drift = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const x = (Math.random() - 0.5) * BOX;
    const y = Math.random() * BOX * 0.6;
    const z = (Math.random() - 0.5) * BOX;
    drift[i] = Math.random() * Math.PI * 2;
    if (weather === 'rain') pos.set([x, y, z, x + 0.05, y + 0.9, z + 0.05], i * 6);
    else pos.set([x, y, z], i * 3);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const obj = weather === 'rain'
    ? new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: '#aebbc6', transparent: true, opacity: 0.45 }))
    : new THREE.Points(geo, new THREE.PointsMaterial({ color: '#ffffff', size: 0.07, map: softDot(), transparent: true, depthWrite: false, opacity: 0.95 }));
  obj.frustumCulled = false;
  scene.add(obj);
  let t = 0;
  const center = new THREE.Vector3();
  return {
    update(dt, camera) {
      t += dt;
      center.copy(camera.position);
      const stride = weather === 'rain' ? 6 : 3;
      const fall = weather === 'rain' ? 26 : 2.2;
      for (let i = 0; i < count; i++) {
        const o = i * stride;
        let x = pos[o] - center.x;
        let y = pos[o + 1] - center.y;
        let z = pos[o + 2] - center.z;
        y -= fall * dt;
        if (weather === 'snow') {
          x += Math.sin(t * 0.8 + drift[i]) * 0.6 * dt;
          z += Math.cos(t * 0.6 + drift[i]) * 0.5 * dt;
        } else {
          x += 2.5 * dt;
        }
        if (y < -12) y += BOX * 0.6;
        if (x > BOX / 2) x -= BOX;
        if (x < -BOX / 2) x += BOX;
        if (z > BOX / 2) z -= BOX;
        if (z < -BOX / 2) z += BOX;
        pos[o] = x + center.x;
        pos[o + 1] = y + center.y;
        pos[o + 2] = z + center.z;
        if (stride === 6) {
          pos[o + 3] = pos[o] - 0.06;
          pos[o + 4] = pos[o + 1] + 0.9;
          pos[o + 5] = pos[o + 2];
        }
      }
      geo.attributes.position.needsUpdate = true;
    },
  };
}

const PARTICLE_VERT = `
attribute float size;
attribute float alpha;
attribute vec3 tint;
varying float vAlpha;
varying vec3 vTint;
void main() {
  vAlpha = alpha;
  vTint = tint;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = size * (420.0 / -mv.z);
  gl_Position = projectionMatrix * mv;
}`;

const PARTICLE_FRAG = `
uniform sampler2D map;
varying float vAlpha;
varying vec3 vTint;
void main() {
  vec4 t = texture2D(map, gl_PointCoord);
  if (t.a * vAlpha < 0.02) discard;
  gl_FragColor = vec4(vTint, t.a * vAlpha);
  #include <colorspace_fragment>
}`;

export function createSpray(scene, quality) {
  const max = Math.floor(3500 * quality.particles) + 400;
  const pos = new Float32Array(max * 3);
  const vel = new Float32Array(max * 3);
  const tint = new Float32Array(max * 3);
  const size = new Float32Array(max);
  const alpha = new Float32Array(max);
  const life = new Float32Array(max);
  const maxLife = new Float32Array(max);
  const kind = new Uint8Array(max);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('tint', new THREE.BufferAttribute(tint, 3));
  geo.setAttribute('size', new THREE.BufferAttribute(size, 1));
  geo.setAttribute('alpha', new THREE.BufferAttribute(alpha, 1));
  const mat = new THREE.ShaderMaterial({ uniforms: { map: { value: softDot() } }, vertexShader: PARTICLE_VERT, fragmentShader: PARTICLE_FRAG, transparent: true, depthWrite: false });
  const points = new THREE.Points(geo, mat);
  points.frustumCulled = false;
  scene.add(points);
  let next = 0;
  const color = new THREE.Color();

  function emit(x, y, z, vx, vy, vz, c, s, l, k) {
    const i = next;
    next = (next + 1) % max;
    pos.set([x, y, z], i * 3);
    vel.set([vx, vy, vz], i * 3);
    color.set(c);
    tint.set([color.r, color.g, color.b], i * 3);
    size[i] = s;
    life[i] = l;
    maxLife[i] = l;
    kind[i] = k;
  }

  return {
    emitFromCar(car, weather, dt, heightAt) {
      const speed = Math.abs(car.u);
      if (car.air || speed < 2) return;
      const s = car.spec;
      const fx = Math.sin(car.heading);
      const fz = Math.cos(car.heading);
      const lx = Math.cos(car.heading);
      const lz = -Math.sin(car.heading);
      const puddle = car.surface === SURFACES.puddle;
      const offroad = car.surface === SURFACES.offroad;
      const throttle = car.input ? car.input.throttle : 0;
      const intensity = Math.min(1, speed / 25) * (0.4 + throttle * 0.6 + car.slip + car.wheelSpin) * (puddle ? 3 : 1);
      const n = Math.floor(intensity * 90 * quality.particles * dt * 10);
      const mud = weather === 'snow' ? '#e9edf0' : puddle ? '#4a3a2a' : offroad ? '#6b5a44' : '#4d3a28';
      for (let k = 0; k < n; k++) {
        const side = k % 2 === 0 ? 1 : -1;
        const wx = car.x + lx * side * s.track / 2 - fx * s.wheelbase / 2;
        const wz = car.z + lz * side * s.track / 2 - fz * s.wheelbase / 2;
        const back = -(3 + Math.random() * 6) * (car.gear === -1 ? -1 : 1);
        emit(wx, heightAt(wx, wz) + 0.3, wz, fx * back + lx * side * (Math.random() * 2.5) + car.vx * 0.3, 2 + Math.random() * (puddle ? 6 : 4), fz * back + lz * side * (Math.random() * 2.5) + car.vz * 0.3, mud, 0.09 + Math.random() * 0.12, 0.6 + Math.random() * 0.6, 0);
      }
      const dusty = weather === 'clear' && (offroad || Math.random() < 0.5);
      if (dusty || weather === 'snow') {
        const d = Math.floor(intensity * 18 * quality.particles * dt * 10);
        for (let k = 0; k < d; k++) {
          const wx = car.x - fx * s.length * 0.55 + (Math.random() - 0.5) * s.width;
          const wz = car.z - fz * s.length * 0.55 + (Math.random() - 0.5) * s.width;
          emit(wx, heightAt(wx, wz) + 0.5, wz, car.vx * 0.2 + (Math.random() - 0.5), 0.6 + Math.random(), car.vz * 0.2 + (Math.random() - 0.5), weather === 'snow' ? '#f4f6f8' : '#b59f7c', 1.4 + Math.random() * 1.6, 1.6 + Math.random() * 1.5, 1);
        }
      }
    },
    burst(x, y, z, strength, weather) {
      const n = Math.min(160, Math.floor(strength * 12 * quality.particles));
      for (let k = 0; k < n; k++) {
        const a = Math.random() * Math.PI * 2;
        const v = 2 + Math.random() * strength;
        emit(x, y + 0.3, z, Math.cos(a) * v, 1 + Math.random() * strength * 0.6, Math.sin(a) * v, weather === 'snow' ? '#eef2f5' : '#57432f', 0.1 + Math.random() * 0.15, 0.7 + Math.random() * 0.5, 0);
      }
    },
    update(dt) {
      for (let i = 0; i < max; i++) {
        if (life[i] <= 0) {
          alpha[i] = 0;
          continue;
        }
        life[i] -= dt;
        const o = i * 3;
        if (kind[i] === 0) {
          vel[o + 1] -= 9.81 * dt;
          alpha[i] = Math.min(1, life[i] * 2);
        } else {
          vel[o] *= 1 - dt * 0.6;
          vel[o + 2] *= 1 - dt * 0.6;
          vel[o + 1] *= 1 - dt * 0.8;
          size[i] += dt * 1.8;
          alpha[i] = (life[i] / maxLife[i]) * 0.32;
        }
        pos[o] += vel[o] * dt;
        pos[o + 1] += vel[o + 1] * dt;
        pos[o + 2] += vel[o + 2] * dt;
      }
      geo.attributes.position.needsUpdate = true;
      geo.attributes.alpha.needsUpdate = true;
      geo.attributes.size.needsUpdate = true;
      geo.attributes.tint.needsUpdate = true;
    },
  };
}

export function createTireMarks(scene, weather) {
  const max = 9000;
  const pos = new Float32Array(max * 4 * 3);
  const col = new Float32Array(max * 4 * 4);
  const idx = new Uint32Array(max * 6);
  for (let q = 0; q < max; q++) idx.set([q * 4, q * 4 + 2, q * 4 + 1, q * 4 + 1, q * 4 + 2, q * 4 + 3], q * 6);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 4));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, transparent: true, depthWrite: false, roughness: weather === 'clear' ? 0.9 : 0.25, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  mesh.receiveShadow = true;
  scene.add(mesh);
  let next = 0;
  const last = new Map();
  const tone = weather === 'snow' ? [0.36, 0.33, 0.3] : [0.16, 0.11, 0.07];

  function quad(a, b, width, strength) {
    const dx = b[0] - a[0];
    const dz = b[2] - a[2];
    const len = Math.hypot(dx, dz) || 1;
    const px = (-dz / len) * width / 2;
    const pz = (dx / len) * width / 2;
    const q = next;
    next = (next + 1) % max;
    pos.set([a[0] - px, a[1], a[2] - pz, a[0] + px, a[1], a[2] + pz, b[0] - px, b[1], b[2] - pz, b[0] + px, b[1], b[2] + pz], q * 12);
    for (let v = 0; v < 4; v++) col.set([...tone, strength], q * 16 + v * 4);
  }

  return {
    update(cars, heightAt) {
      let dirty = false;
      cars.forEach((car, k) => {
        const s = car.spec;
        const fx = Math.sin(car.heading);
        const fz = Math.cos(car.heading);
        const lx = Math.cos(car.heading);
        const lz = -Math.sin(car.heading);
        [1, -1].forEach((side) => {
          const key = k * 2 + (side > 0 ? 0 : 1);
          const x = car.x + lx * side * s.track / 2 - fx * s.wheelbase / 2;
          const z = car.z + lz * side * s.track / 2 - fz * s.wheelbase / 2;
          const p = [x, heightAt(x, z) + ROAD_OFFSET + 0.015, z];
          const prev = last.get(key);
          if (car.air) {
            last.delete(key);
            return;
          }
          if (!prev) {
            last.set(key, p);
            return;
          }
          const d = Math.hypot(p[0] - prev[0], p[2] - prev[2]);
          if (d < 0.6) return;
          if (d < 4) {
            quad(prev, p, s.wheelR * 0.62, Math.min(0.75, 0.32 + car.slip * 0.5 + car.wheelSpin * 0.3));
            dirty = true;
          }
          last.set(key, p);
        });
      });
      if (dirty) {
        geo.attributes.position.needsUpdate = true;
        geo.attributes.color.needsUpdate = true;
      }
    },
  };
}
