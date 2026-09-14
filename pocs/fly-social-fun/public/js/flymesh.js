import * as THREE from 'three';

const MODEL_SCALE = 1.5;
const UP = new THREE.Vector3(0, 1, 0);

const geometries = {
  thorax: new THREE.SphereGeometry(0.16, 20, 16),
  abdomen: new THREE.SphereGeometry(0.17, 20, 16),
  head: new THREE.SphereGeometry(0.12, 20, 16),
  eye: new THREE.SphereGeometry(0.085, 16, 12),
  shine: new THREE.SphereGeometry(0.022, 8, 6),
  wing: wingGeometry(),
  leg: new THREE.CylinderGeometry(0.012, 0.009, 1, 5),
  hit: new THREE.SphereGeometry(0.45, 8, 6),
  cocoon: new THREE.SphereGeometry(0.34, 14, 10),
};

const materialCache = new Map();
const shineMaterial = new THREE.MeshBasicMaterial({ color: '#ffffff' });
const hitMaterial = new THREE.MeshBasicMaterial({ visible: false });
const cocoonMaterial = new THREE.MeshStandardMaterial({ color: '#f4f1ea', roughness: 1 });

function wingGeometry() {
  const shape = new THREE.Shape();
  shape.absellipse(0, 0, 0.1, 0.3, 0, Math.PI * 2);
  const geometry = new THREE.ShapeGeometry(shape, 24);
  geometry.rotateX(-Math.PI / 2);
  geometry.translate(0, 0, -0.26);
  return geometry;
}

function materialsFor(species) {
  if (!materialCache.has(species.label)) {
    materialCache.set(species.label, {
      body: new THREE.MeshStandardMaterial({ color: species.body, roughness: 0.45, metalness: 0.35 }),
      eyes: new THREE.MeshStandardMaterial({ color: species.eyes, roughness: 0.25, metalness: 0.1 }),
      wings: new THREE.MeshPhysicalMaterial({
        color: species.wings,
        transparent: true,
        opacity: 0.5,
        roughness: 0.15,
        iridescence: 1,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    });
  }
  return materialCache.get(species.label);
}

function part(geometry, material, position, scale) {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...position);
  if (scale) mesh.scale.set(...scale);
  mesh.castShadow = true;
  return mesh;
}

function segment(material, from, to) {
  const start = new THREE.Vector3(...from);
  const direction = new THREE.Vector3(...to).sub(start);
  const mesh = new THREE.Mesh(geometries.leg, material);
  mesh.scale.set(1, direction.length(), 1);
  mesh.position.copy(start).addScaledVector(direction, 0.5);
  mesh.quaternion.setFromUnitVectors(UP, direction.normalize());
  return mesh;
}

function buildLeg(material, side, z, reach) {
  const leg = new THREE.Group();
  leg.position.set(0.09 * side, -0.08, z);
  const knee = [0.16 * side, 0.02, reach * 0.5];
  leg.add(segment(material, [0, 0, 0], knee));
  leg.add(segment(material, knee, [0.24 * side, -0.2, reach]));
  return leg;
}

function buildWing(material, side) {
  const pivot = new THREE.Group();
  pivot.position.set(0.07 * side, 0.13, 0.02);
  const wing = new THREE.Mesh(geometries.wing, material);
  wing.rotation.y = 0.45 * side;
  pivot.add(wing);
  return pivot;
}

export function buildFlyMesh(species) {
  const materials = materialsFor(species);
  const group = new THREE.Group();
  const body = new THREE.Group();
  group.add(body);

  body.add(part(geometries.thorax, materials.body, [0, 0, 0.04], [1, 0.95, 1.1]));
  body.add(part(geometries.abdomen, materials.body, [0, -0.02, -0.26], [0.95, 0.8, 1.5]));
  body.add(part(geometries.head, materials.body, [0, 0.02, 0.26]));
  for (const side of [-1, 1]) {
    body.add(part(geometries.eye, materials.eyes, [0.075 * side, 0.05, 0.3]));
    body.add(part(geometries.shine, shineMaterial, [0.1 * side, 0.1, 0.36]));
  }

  const wings = [buildWing(materials.wings, -1), buildWing(materials.wings, 1)];
  wings.forEach((wing) => body.add(wing));

  const legs = [];
  for (const side of [-1, 1]) {
    legs.push(buildLeg(materials.body, side, 0.12, 0.14));
    legs.push(buildLeg(materials.body, side, 0.03, 0));
    legs.push(buildLeg(materials.body, side, -0.07, -0.16));
  }
  legs.forEach((leg) => body.add(leg));

  const hit = new THREE.Mesh(geometries.hit, hitMaterial);
  group.add(hit);
  group.scale.setScalar(species.size * MODEL_SCALE);

  return { group, body, wings, frontLegs: [legs[0], legs[3]], hit };
}

export function buildCocoon() {
  const cocoon = new THREE.Mesh(geometries.cocoon, cocoonMaterial);
  cocoon.scale.set(1, 1, 1.5);
  cocoon.position.z = -0.05;
  return cocoon;
}
