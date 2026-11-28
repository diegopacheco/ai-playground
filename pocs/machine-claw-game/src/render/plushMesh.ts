import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { PLUSH_LOOKS, type PlushLook } from "../game/plush";

const BALL = new THREE.SphereGeometry(1, 24, 16);
const BEAD = new THREE.SphereGeometry(1, 12, 8);
const CONE = new THREE.ConeGeometry(1, 1, 14);
const EYE = new THREE.MeshPhysicalMaterial({ color: 0x14101c, roughness: 0.05, clearcoat: 1, clearcoatRoughness: 0.05 });
const SHINE = new THREE.MeshBasicMaterial({ color: 0xffffff });
const BLUSH = new THREE.MeshStandardMaterial({ color: 0xff8fb4, roughness: 0.9, transparent: true, opacity: 0.85 });
const BEAK = new THREE.MeshStandardMaterial({ color: 0xffa62b, roughness: 0.6 });
const GOLD = new THREE.MeshStandardMaterial({ color: 0xffd35c, roughness: 0.3, metalness: 0.4 });
const SPIKE = new THREE.MeshStandardMaterial({ color: 0xffb347, roughness: 0.8 });
const MANE = new THREE.MeshStandardMaterial({ color: 0xc9a4ff, roughness: 0.85 });
const furs = new Map<number, THREE.MeshPhysicalMaterial>();
const built = new Map<number, readonly (readonly [THREE.BufferGeometry, THREE.Material])[]>();

function fur(color: number): THREE.MeshPhysicalMaterial {
  const known = furs.get(color);
  if (known) return known;
  const made = new THREE.MeshPhysicalMaterial({
    color,
    roughness: 0.95,
    sheen: 0.6,
    sheenRoughness: 0.45,
    sheenColor: new THREE.Color(color).lerp(new THREE.Color(0xffffff), 0.3),
  });
  furs.set(color, made);
  return made;
}

type Vec = readonly [x: number, y: number, z: number];

type Part = {
  readonly material: THREE.Material;
  readonly shape: THREE.BufferGeometry;
  readonly size: Vec;
  readonly at: Vec;
  readonly turn?: Vec;
};

function ball(material: THREE.Material, size: Vec, at: Vec, turn?: Vec): Part {
  return { material, shape: BALL, size, at, turn };
}

function bead(material: THREE.Material, size: Vec, at: Vec): Part {
  return { material, shape: BEAD, size, at };
}

function cone(material: THREE.Material, size: Vec, at: Vec, turn: Vec): Part {
  return { material, shape: CONE, size, at, turn };
}

const SIDES = [-1, 1] as const;

function ears(look: PlushLook): readonly Part[] {
  const trim = fur(look.trim);
  const inner = fur(look.accent);
  switch (look.ears) {
    case "round":
      return SIDES.flatMap((side) => [
        ball(trim, [0.3, 0.3, 0.16], [side * 0.62, 0.78, 0]),
        ball(inner, [0.17, 0.17, 0.08], [side * 0.62, 0.78, 0.1]),
      ]);
    case "long":
      return SIDES.flatMap((side) => [
        ball(trim, [0.17, 0.5, 0.11], [side * 0.34, 1.2, 0]),
        ball(inner, [0.09, 0.36, 0.05], [side * 0.34, 1.2, 0.08]),
      ]);
    case "pointy":
      return SIDES.map((side) => cone(trim, [0.26, 0.42, 0.16], [side * 0.52, 0.98, 0], [0, 0, -side * 0.35]));
    case "none":
      return [];
    default: {
      const unreachable: never = look.ears;
      return unreachable;
    }
  }
}

function face(look: PlushLook): readonly Part[] {
  const accent = fur(look.accent);
  const snout =
    look.extra === "beak"
      ? [cone(BEAK, [0.13, 0.22, 0.1], [0, 0.08, 1.02], [Math.PI / 2, 0, 0])]
      : [ball(accent, [0.27, 0.19, 0.16], [0, 0.06, 0.9]), bead(EYE, [0.07, 0.05, 0.04], [0, 0.15, 1.04])];
  return [
    ...snout,
    ...SIDES.flatMap((side) => [
      ball(EYE, [0.12, 0.14, 0.08], [side * 0.33, 0.3, 0.86]),
      bead(SHINE, [0.045, 0.045, 0.02], [side * 0.33 - 0.035, 0.35, 0.935]),
      bead(SHINE, [0.02, 0.02, 0.015], [side * 0.33 + 0.03, 0.25, 0.935]),
      bead(BLUSH, [0.13, 0.075, 0.04], [side * 0.56, 0.08, 0.8]),
    ]),
  ];
}

function limbs(look: PlushLook): readonly Part[] {
  const trim = fur(look.trim);
  const coat = fur(look.fur);
  if (look.extra === "tentacles") {
    return Array.from({ length: 6 }, (_, index) => {
      const angle = (index / 6) * Math.PI * 2 + Math.PI / 6;
      return ball(coat, [0.24, 0.16, 0.24], [Math.cos(angle) * 0.62, -0.82, Math.sin(angle) * 0.62]);
    });
  }
  const arm: Vec = look.extra === "beak" ? [0.12, 0.38, 0.22] : [0.2, 0.28, 0.2];
  return SIDES.flatMap((side) => [
    ball(trim, [0.24, 0.2, 0.3], [side * 0.45, -0.86, 0.38]),
    ball(trim, arm, [side * 0.93, -0.12, 0.3], [0, 0, side * 0.3]),
  ]);
}

function extras(look: PlushLook): readonly Part[] {
  const trim = fur(look.trim);
  const accent = fur(look.accent);
  switch (look.extra) {
    case "spikes":
      return [
        ...[0.25, 0.7, 1.15, 1.6].map((angle) =>
          cone(SPIKE, [0.13, 0.24, 0.09], [0, Math.cos(angle) * 0.98, -Math.sin(angle) * 0.98], [-angle, 0, 0]),
        ),
        cone(trim, [0.22, 0.6, 0.22], [0, -0.62, -0.95], [-2.1, 0, 0]),
      ];
    case "horn":
      return [
        cone(GOLD, [0.1, 0.42, 0.1], [0, 1.12, 0.3], [0.35, 0, 0]),
        ...[0.4, 0.85, 1.3].map((angle) => ball(MANE, [0.22, 0.2, 0.2], [0, Math.cos(angle) * 0.95, -Math.sin(angle) * 0.95])),
      ];
    case "tail":
      return [
        ball(trim, [0.26, 0.26, 0.5], [0.42, -0.5, -0.9], [0.5, 0.5, 0]),
        ball(accent, [0.16, 0.16, 0.2], [0.58, -0.36, -1.28]),
      ];
    case "fin":
      return SIDES.map((side) => ball(trim, [0.34, 0.1, 0.2], [side * 0.26, 0.5, -1.02], [0.4, 0, side * 0.5]));
    case "beak":
      return [ball(accent, [0.6, 0.66, 0.3], [0, -0.18, 0.72])];
    case "tentacles":
    case "none":
      return [];
    default: {
      const unreachable: never = look.extra;
      return unreachable;
    }
  }
}

function parts(look: PlushLook): readonly Part[] {
  const belly: readonly Part[] = look.extra === "beak" ? [] : [ball(fur(look.accent), [0.5, 0.42, 0.2], [0, -0.3, 0.8])];
  return [ball(fur(look.fur), [1, 0.97, 1], [0, 0, 0]), ...belly, ...face(look), ...limbs(look), ...ears(look), ...extras(look)];
}

function placed(part: Part): THREE.BufferGeometry {
  const turn = part.turn ?? [0, 0, 0];
  const matrix = new THREE.Matrix4().compose(
    new THREE.Vector3(...part.at),
    new THREE.Quaternion().setFromEuler(new THREE.Euler(...turn)),
    new THREE.Vector3(...part.size),
  );
  return part.shape.clone().applyMatrix4(matrix);
}

function meshesFor(index: number, look: PlushLook): readonly (readonly [THREE.BufferGeometry, THREE.Material])[] {
  const known = built.get(index);
  if (known) return known;
  const byMaterial = new Map<THREE.Material, THREE.BufferGeometry[]>();
  for (const part of parts(look)) {
    const list = byMaterial.get(part.material) ?? [];
    list.push(placed(part));
    byMaterial.set(part.material, list);
  }
  const made = [...byMaterial].map(([material, pieces]) => {
    const merged = mergeGeometries(pieces);
    if (!merged) throw new Error(`could not build plush ${look.name}`);
    for (const piece of pieces) piece.dispose();
    return [merged, material] as const;
  });
  built.set(index, made);
  return made;
}

export function createPlush(lookIndex: number, radius: number): THREE.Group {
  const index = lookIndex % PLUSH_LOOKS.length;
  const look = PLUSH_LOOKS[index];
  if (!look) throw new Error(`unknown plush look ${lookIndex}`);
  const group = new THREE.Group();
  for (const [geometry, material] of meshesFor(index, look)) {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = material !== SHINE && material !== BLUSH;
    mesh.receiveShadow = true;
    group.add(mesh);
  }
  group.scale.setScalar(radius);
  return group;
}
