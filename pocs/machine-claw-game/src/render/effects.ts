import * as THREE from "three";

export type Confetti = {
  readonly points: THREE.Points;
  burst(at: THREE.Vector3): void;
  update(dt: number): void;
};

export type CoinToss = {
  readonly mesh: THREE.Mesh;
  toss(): void;
  update(dt: number): void;
};

const PIECES = 220;
const LIFE = 2.6;

export function createConfetti(): Confetti {
  const positions = new Float32Array(PIECES * 3);
  const colors = new Float32Array(PIECES * 3);
  const velocities = new Float32Array(PIECES * 3);
  const geometry = new THREE.BufferGeometry();
  const positionAttribute = new THREE.BufferAttribute(positions, 3);
  const colorAttribute = new THREE.BufferAttribute(colors, 3);
  geometry.setAttribute("position", positionAttribute);
  geometry.setAttribute("color", colorAttribute);
  const material = new THREE.PointsMaterial({ size: 0.035, vertexColors: true, transparent: true, depthWrite: false });
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  points.visible = false;
  const tint = new THREE.Color();
  let age = LIFE;

  function burst(at: THREE.Vector3): void {
    for (let i = 0; i < PIECES; i += 1) {
      const turn = Math.random() * Math.PI * 2;
      const speed = 0.4 + Math.random() * 1.4;
      positions.set([at.x, at.y, at.z], i * 3);
      velocities.set([Math.cos(turn) * speed, 1.2 + Math.random() * 2.2, Math.sin(turn) * speed * 0.6], i * 3);
      tint.setHSL(Math.random(), 1, 0.6);
      colors.set([tint.r, tint.g, tint.b], i * 3);
    }
    colorAttribute.needsUpdate = true;
    age = 0;
    points.visible = true;
  }

  function update(dt: number): void {
    if (age >= LIFE) return;
    age += dt;
    for (let i = 0; i < PIECES * 3; i += 3) {
      velocities[i + 1] = (velocities[i + 1] ?? 0) - 3.2 * dt;
      positions[i] = (positions[i] ?? 0) + (velocities[i] ?? 0) * dt;
      positions[i + 1] = (positions[i + 1] ?? 0) + (velocities[i + 1] ?? 0) * dt;
      positions[i + 2] = (positions[i + 2] ?? 0) + (velocities[i + 2] ?? 0) * dt;
    }
    positionAttribute.needsUpdate = true;
    material.opacity = Math.max(0, 1 - age / LIFE);
    points.visible = age < LIFE;
  }

  return { points, burst, update };
}

const TOSS_TIME = 0.55;

export function createCoinToss(slot: THREE.Vector3): CoinToss {
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(0.04, 0.04, 0.007, 28),
    new THREE.MeshStandardMaterial({ color: 0xffc83d, roughness: 0.22, metalness: 1 }),
  );
  mesh.visible = false;
  const from = new THREE.Vector3(slot.x + 0.25, slot.y + 0.45, slot.z + 0.7);
  let age = TOSS_TIME;

  function update(dt: number): void {
    if (age >= TOSS_TIME) return;
    age += dt;
    const progress = Math.min(1, age / TOSS_TIME);
    mesh.position.lerpVectors(from, slot, progress);
    mesh.position.y += Math.sin(progress * Math.PI) * 0.18;
    mesh.rotation.set(Math.PI / 2, progress * 14, 0);
    mesh.visible = progress < 1;
  }

  return {
    mesh,
    toss: () => {
      age = 0;
    },
    update,
  };
}
