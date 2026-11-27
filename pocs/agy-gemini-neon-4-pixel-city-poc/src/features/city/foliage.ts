import * as THREE from "three";
import { LANDMARKS } from "./catalog";
import { PALETTE } from "./palette";
import { createOutlinedBox, createOutlinedCylinder } from "./geometry";
import { createPinSpriteTexture } from "./textures";

export type FoliageRig = {
  readonly group: THREE.Group;
  readonly lampBulbs: THREE.MeshBasicMaterial[];
  readonly pinSprites: { readonly sprite: THREE.Sprite; readonly baseY: number; readonly phase: number }[];
  readonly pinsGroup: THREE.Group;
};

function createLeafTree(x: number, z: number, scale = 1): THREE.Group {
  const tree = new THREE.Group();
  const trunk = createOutlinedCylinder(0.11 * scale, 0.14 * scale, 0.65 * scale, 6, PALETTE.trunk);
  trunk.position.y = 0.32 * scale;
  tree.add(trunk);

  const low = createOutlinedBox({
    width: 0.92 * scale,
    height: 0.56 * scale,
    depth: 0.92 * scale,
    color: PALETTE.treeLeafDark,
  });
  low.position.y = 0.82 * scale;
  tree.add(low);

  const top = createOutlinedBox({
    width: 0.66 * scale,
    height: 0.44 * scale,
    depth: 0.66 * scale,
    color: PALETTE.treeLeafLight,
  });
  top.position.y = 1.24 * scale;
  tree.add(top);

  tree.position.set(x, 0.04, z);
  return tree;
}

function createPalmTree(x: number, z: number, height = 1.8): THREE.Group {
  const palm = new THREE.Group();
  const trunk = createOutlinedCylinder(0.08, 0.14, height, 6, PALETTE.trunk);
  trunk.position.y = height / 2;
  trunk.rotation.z = 0.08;
  palm.add(trunk);

  for (let i = 0; i < 5; i++) {
    const angle = (i / 5) * Math.PI * 2;
    const frond = createOutlinedBox({
      width: 0.95,
      height: 0.08,
      depth: 0.24,
      color: PALETTE.palmLeaf,
    });
    frond.position.set(Math.cos(angle) * 0.45, height + 0.04, Math.sin(angle) * 0.45);
    frond.rotation.y = -angle;
    frond.rotation.z = -0.28;
    palm.add(frond);
  }

  palm.position.set(x, 0.04, z);
  return palm;
}

function createBench(x: number, z: number, rotateY: number): THREE.Group {
  const bench = new THREE.Group();
  const seat = createOutlinedBox({
    width: 0.65,
    height: 0.08,
    depth: 0.24,
    color: PALETTE.pierWood,
  });
  seat.position.y = 0.2;
  bench.add(seat);

  const back = createOutlinedBox({
    width: 0.65,
    height: 0.22,
    depth: 0.06,
    color: PALETTE.pierWood,
  });
  back.position.set(0, 0.34, -0.1);
  bench.add(back);

  bench.position.set(x, 0.04, z);
  bench.rotation.y = rotateY;
  return bench;
}

function createStreetLamp(
  x: number,
  z: number,
  lampBulbs: THREE.MeshBasicMaterial[],
): THREE.Group {
  const lamp = new THREE.Group();
  const pole = createOutlinedCylinder(0.04, 0.06, 1.35, 6, PALETTE.ink, false);
  pole.position.y = 0.68;
  lamp.add(pole);

  const cap = createOutlinedBox({
    width: 0.22,
    height: 0.08,
    depth: 0.22,
    color: PALETTE.ink,
    outline: false,
  });
  cap.position.y = 1.46;
  lamp.add(cap);

  const bulbMat = new THREE.MeshBasicMaterial({ color: PALETTE.lampGlow });
  lampBulbs.push(bulbMat);
  const bulb = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.16, 0.16), bulbMat);
  bulb.position.y = 1.35;
  lamp.add(bulb);

  lamp.position.set(x, 0.04, z);
  return lamp;
}

export function buildFoliageAndProps(): FoliageRig {
  const group = new THREE.Group();
  const lampBulbs: THREE.MeshBasicMaterial[] = [];

  const parkTrees: readonly (readonly [number, number, number])[] = [
    [-4.8, -1.8, 1.05],
    [-4.4, -0.4, 0.95],
    [-4.9, 1.0, 1.1],
    [-3.7, 2.0, 0.95],
    [-2.4, 2.3, 1.0],
    [-1.4, 0.6, 0.9],
    [-3.9, -2.5, 0.9],
    [-8.3, 10.6, 0.85],
    [-3.1, 10.6, 0.9],
    [-7.9, 8.9, 0.9],
  ];
  for (const [x, z, s] of parkTrees) {
    group.add(createLeafTree(x, z, s));
  }

  const palms: readonly (readonly [number, number, number])[] = [
    [-10.8, -10.8, 1.9],
    [-9.7, -11.0, 1.65],
    [-10.9, -9.6, 1.75],
    [1.8, -9.8, 1.7],
    [2.3, -7.2, 1.65],
    [2.3, -4.4, 1.75],
    [0.8, -1.3, 1.55],
    [2.1, -1.2, 1.6],
  ];
  for (const [x, z, h] of palms) {
    group.add(createPalmTree(x, z, h));
  }

  const benches: readonly (readonly [number, number, number])[] = [
    [-3.9, 0.8, Math.PI / 2],
    [-2.3, 0.9, -Math.PI / 2],
    [0.8, -5.6, Math.PI / 2],
    [2.8, -9.2, 0],
  ];
  for (const [x, z, r] of benches) {
    group.add(createBench(x, z, r));
  }

  const lampSpots: readonly (readonly [number, number])[] = [
    [-5.6, 9.2],
    [-5.6, 6.2],
    [-5.6, 2.8],
    [-2.8, 6.1],
    [1.2, 6.1],
    [5.2, 6.1],
    [8.2, 6.1],
    [1.8, 10.9],
    [5.6, 10.9],
    [8.4, 10.9],
    [10.8, 7.2],
    [10.8, 3.0],
    [10.8, -1.2],
    [0.1, -8.2],
    [2.5, -2.9],
  ];
  for (const [x, z] of lampSpots) {
    group.add(createStreetLamp(x, z, lampBulbs));
  }

  const pinsGroup = new THREE.Group();
  const pinSprites: { sprite: THREE.Sprite; baseY: number; phase: number }[] = [];

  LANDMARKS.forEach((landmark, idx) => {
    const tex = createPinSpriteTexture(landmark.pinKind);
    const mat = new THREE.SpriteMaterial({
      map: tex,
      transparent: true,
      depthTest: false,
    });
    const sprite = new THREE.Sprite(mat);
    const baseY = landmark.position[1] + 1.15;
    sprite.position.set(landmark.position[0], baseY, landmark.position[2]);
    sprite.scale.set(1.25, 1.25, 1);
    sprite.renderOrder = 10;
    sprite.userData.landmarkId = landmark.id;
    pinsGroup.add(sprite);
    pinSprites.push({ sprite, baseY, phase: idx * 0.95 });
  });

  group.add(pinsGroup);
  return { group, lampBulbs, pinSprites, pinsGroup };
}
