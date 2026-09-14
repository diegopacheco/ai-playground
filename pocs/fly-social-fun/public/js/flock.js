import * as THREE from 'three';
import { buildFlyMesh, buildCocoon } from './flymesh.js';

const BOUNDS = { minX: -12.6, maxX: 12.6, minY: 0.35, maxY: 13, minZ: -7.5, maxZ: 7.6 };
const GROUND_Y = 0.3;
const HOVER_ONLY = new Set(['lamp', 'window', 'web']);
const MAX_SPEED = 7.5;
const CORPSE_SECONDS = 30;
const FADE_SECONDS = 1.5;

function heartTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ff4f86';
  ctx.beginPath();
  ctx.moveTo(32, 56);
  ctx.bezierCurveTo(4, 36, 4, 10, 20, 10);
  ctx.bezierCurveTo(28, 10, 32, 18, 32, 20);
  ctx.bezierCurveTo(32, 18, 36, 10, 44, 10);
  ctx.bezierCurveTo(60, 10, 60, 36, 32, 56);
  ctx.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function bezier(out, from, to, lift, t) {
  const mx = (from.x + to.x) / 2;
  const my = Math.max(from.y, to.y) + lift;
  const mz = (from.z + to.z) / 2;
  const u = 1 - t;
  return out.set(
    u * u * from.x + 2 * u * t * mx + t * t * to.x,
    u * u * from.y + 2 * u * t * my + t * t * to.y,
    u * u * from.z + 2 * u * t * mz + t * t * to.z,
  );
}

export function createFlock(stage) {
  const flies = new Map();
  const dummy = new THREE.Object3D();
  const toAnchor = new THREE.Vector3();
  const anchor = new THREE.Vector3();
  const forward = new THREE.Vector3();
  const heart = heartTexture();
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.7, 0.045, 8, 40),
    new THREE.MeshBasicMaterial({ color: '#b8ff5c', transparent: true, opacity: 0.9 }),
  );
  ring.visible = false;
  stage.scene.add(ring);
  let species = {};
  let selectedId = null;

  function around(spotId, spread) {
    return stage.spotPosition(spotId).clone().add(
      new THREE.Vector3((Math.random() - 0.5) * spread, 0.6 + Math.random() * spread * 0.5, (Math.random() - 0.5) * spread),
    );
  }

  function spawn(data, position, velocity) {
    const mesh = buildFlyMesh(species[data.species]);
    mesh.hit.userData.flyId = data.id;
    const fly = {
      id: data.id,
      mesh,
      spot: data.spot,
      pos: position.clone(),
      vel: velocity ? velocity.clone() : new THREE.Vector3(),
      jitter: new THREE.Vector3(),
      jitterTimer: 0,
      mode: 'fly',
      rest: 0,
      landPoint: new THREE.Vector3(),
      orbit: {
        radius: 0.7 + Math.random() * 1.1,
        speed: (0.5 + Math.random() * 0.9) * (Math.random() < 0.5 ? -1 : 1),
        phase: Math.random() * Math.PI * 2,
      },
      flapPhase: Math.random() * 10,
      dead: null,
    };
    mesh.group.position.copy(fly.pos);
    stage.scene.add(mesh.group);
    flies.set(fly.id, fly);
    return fly;
  }

  function remove(fly) {
    stage.scene.remove(fly.mesh.group);
    flies.delete(fly.id);
    if (selectedId === fly.id) select(null);
  }

  function sync(list, speciesTable) {
    species = speciesTable;
    const incoming = new Map(list.map((data) => [data.id, data]));
    for (const fly of [...flies.values()]) {
      if (!fly.dead && !incoming.has(fly.id)) remove(fly);
    }
    for (const data of list) {
      if (flies.has(data.id)) moveTo(data.id, data.spot);
      else spawn(data, around(data.spot, 2.5));
    }
  }

  function add(data) {
    const start = stage.spotPosition(data.spot).clone().add(new THREE.Vector3(0, -0.8, 0));
    spawn(data, start, new THREE.Vector3((Math.random() - 0.5) * 3, 7, (Math.random() - 0.5) * 3));
    stage.sparkle(start);
  }

  function moveTo(flyId, spot) {
    const fly = flies.get(flyId);
    if (!fly || fly.dead) return;
    fly.spot = spot;
    fly.mode = 'fly';
  }

  function kill(flyId, cause) {
    const fly = flies.get(flyId);
    if (!fly || fly.dead) return;
    fly.dead = { cause, timer: 0, grounded: fly.mode === 'rest' };
    fly.mode = 'dead';
    if (cause === 'spider') {
      fly.mesh.body.add(buildCocoon());
      fly.mesh.wings.forEach((wing) => (wing.visible = false));
    }
    if (cause === 'swatter') {
      fly.pos.y = Math.max(GROUND_Y * 0.4, stage.spotPosition(fly.spot).y - 0.1);
      fly.mesh.group.scale.multiply(new THREE.Vector3(1.35, 0.22, 1.35));
      fly.dead.grounded = true;
    }
    fly.vel.set(fly.vel.x * 0.2, 0, fly.vel.z * 0.2);
  }

  function steer(fly, dt, time) {
    const base = stage.spotPosition(fly.spot);
    const landing = fly.mode === 'landing';
    if (landing) {
      anchor.copy(fly.landPoint);
    } else {
      const angle = time * fly.orbit.speed + fly.orbit.phase;
      const radius = fly.orbit.radius * (fly.spot === 'web' ? 0.45 : 1);
      anchor.set(
        base.x + Math.cos(angle) * radius + Math.sin(angle * 2.3) * 0.3,
        base.y + 0.9 + Math.sin(angle * 1.7) * 0.45,
        base.z + Math.sin(angle) * radius,
      );
    }
    toAnchor.subVectors(anchor, fly.pos);
    const distance = toAnchor.length();

    fly.jitterTimer -= dt;
    if (fly.jitterTimer <= 0) {
      fly.jitter.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize().multiplyScalar(landing ? 1.5 : 11);
      fly.jitterTimer = 0.12 + Math.random() * 0.28;
    }

    toAnchor.normalize().multiplyScalar(Math.min(distance, 3) * 7);
    fly.vel.addScaledVector(toAnchor.add(fly.jitter).addScaledVector(fly.vel, -1.8), dt);
    fly.vel.clampLength(0, landing ? Math.max(1.2, distance * 3) : MAX_SPEED);
    fly.pos.addScaledVector(fly.vel, dt);
    fly.pos.set(
      THREE.MathUtils.clamp(fly.pos.x, BOUNDS.minX, BOUNDS.maxX),
      THREE.MathUtils.clamp(fly.pos.y, BOUNDS.minY, BOUNDS.maxY),
      THREE.MathUtils.clamp(fly.pos.z, BOUNDS.minZ, BOUNDS.maxZ),
    );

    if (landing && distance < 0.15) {
      fly.mode = 'rest';
      fly.rest = 2.5 + Math.random() * 5;
      fly.pos.copy(fly.landPoint);
      fly.vel.set(0, 0, 0);
    } else if (!landing && !HOVER_ONLY.has(fly.spot) && fly.pos.distanceTo(base) < 2.2 && Math.random() < dt * 0.15) {
      fly.mode = 'landing';
      fly.landPoint.set(base.x + (Math.random() - 0.5) * 0.9, base.y, base.z + (Math.random() - 0.5) * 0.9);
    }
  }

  function flap(fly, time, flying) {
    const lift = flying ? 0.2 + Math.sin(time * 62 + fly.flapPhase) * 0.7 : 0.05;
    const sweep = flying ? 1.05 : 0.3;
    fly.mesh.wings.forEach((pivot, index) => {
      const side = index === 0 ? -1 : 1;
      pivot.rotation.z = side * lift;
      pivot.children[0].rotation.y = -side * sweep;
    });
  }

  function face(fly, dt, direction) {
    if (direction.lengthSq() < 0.0001) return;
    dummy.position.copy(fly.pos);
    dummy.lookAt(forward.copy(fly.pos).add(direction));
    fly.mesh.group.quaternion.slerp(dummy.quaternion, 1 - Math.exp(-8 * dt));
  }

  function rubHands(fly, time, amount) {
    fly.mesh.frontLegs.forEach((leg, index) => {
      leg.rotation.x = Math.sin(time * 14 + index * Math.PI) * amount;
    });
  }

  function updateDead(fly, dt, time) {
    const dead = fly.dead;
    dead.timer += dt;
    if (!dead.grounded && dead.cause !== 'spider') {
      fly.vel.y -= 20 * dt;
      fly.pos.addScaledVector(fly.vel, dt);
      if (fly.pos.y <= GROUND_Y) {
        fly.pos.y = GROUND_Y;
        dead.grounded = true;
      }
    }
    if (dead.cause === 'age') {
      fly.mesh.body.rotation.z = THREE.MathUtils.lerp(fly.mesh.body.rotation.z, Math.PI, 1 - Math.exp(-4 * dt));
      rubHands(fly, time * 2, dead.timer < 3 ? 0.5 : 0);
    }
    if (dead.cause === 'spider') fly.mesh.body.rotation.z = Math.sin(time * 3) * 0.1;
    if (dead.timer > CORPSE_SECONDS) {
      const fade = 1 - (dead.timer - CORPSE_SECONDS) / FADE_SECONDS;
      if (fade <= 0) return remove(fly);
      fly.mesh.body.scale.setScalar(fade);
    }
  }

  function update(dt, time) {
    for (const fly of [...flies.values()]) {
      if (fly.mode === 'dead') {
        updateDead(fly, dt, time);
        flap(fly, time, false);
      } else if (fly.mode === 'rest') {
        fly.rest -= dt;
        rubHands(fly, time, 0.35);
        flap(fly, time, false);
        forward.set(0, 0, 1).applyQuaternion(fly.mesh.group.quaternion).setY(0);
        face(fly, dt, forward.clone());
        if (fly.rest <= 0) {
          fly.mode = 'fly';
          fly.vel.set(0, 3, 0);
        }
      } else {
        steer(fly, dt, time);
        rubHands(fly, time, 0);
        flap(fly, time, true);
        face(fly, dt, fly.vel);
      }
      fly.mesh.group.position.copy(fly.pos);
    }
    const selected = flies.get(selectedId);
    ring.visible = Boolean(selected);
    if (selected) {
      ring.position.copy(selected.pos);
      ring.rotation.set(Math.PI / 2 + Math.sin(time * 2) * 0.2, 0, time * 2);
    }
  }

  function heartTrail(fromId, toId) {
    const from = flies.get(fromId);
    const to = flies.get(toId);
    if (!from || !to) return;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: heart, transparent: true, depthWrite: false }));
    stage.scene.add(sprite);
    let t = 0;
    stage.addEffect((dt) => {
      t += dt / 1.1;
      bezier(sprite.position, from.pos, to.pos, 1.6, Math.min(t, 1));
      sprite.scale.setScalar(0.55 * (1 + Math.sin(Math.min(t, 1) * Math.PI) * 0.5));
      sprite.material.opacity = t > 0.8 ? (1 - t) * 5 : 1;
      if (t < 1) return true;
      stage.scene.remove(sprite);
      sprite.material.dispose();
      return false;
    });
  }

  function followLink(fromId, toId) {
    const from = flies.get(fromId);
    const to = flies.get(toId);
    if (!from || !to) return;
    const points = 24;
    const geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(points * 3), 3));
    const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: '#b8ff5c', transparent: true }));
    stage.scene.add(line);
    const point = new THREE.Vector3();
    let t = 0;
    stage.addEffect((dt) => {
      t += dt / 1.8;
      const attribute = geometry.getAttribute('position');
      for (let i = 0; i < points; i++) {
        bezier(point, from.pos, to.pos, 1.2, i / (points - 1));
        attribute.setXYZ(i, point.x, point.y, point.z);
      }
      attribute.needsUpdate = true;
      line.material.opacity = 1 - t;
      if (t < 1) return true;
      stage.scene.remove(line);
      geometry.dispose();
      line.material.dispose();
      return false;
    });
  }

  function pick(raycaster) {
    const targets = [...flies.values()].filter((fly) => !fly.dead).map((fly) => fly.mesh.hit);
    const [hit] = raycaster.intersectObjects(targets, false);
    return hit ? hit.object.userData.flyId : null;
  }

  function select(flyId) {
    selectedId = flyId;
  }

  function position(flyId) {
    return flies.get(flyId)?.pos || null;
  }

  return { sync, add, moveTo, kill, update, heartTrail, followLink, pick, select, position };
}
