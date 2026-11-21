import * as THREE from "three";
import { PLUSH_LOOKS, type PlushLook } from "../game/plush";

const BALL = new THREE.SphereGeometry(1, 28, 20);
const CONE = new THREE.ConeGeometry(1, 1, 16);
const EYE = new THREE.MeshStandardMaterial({ color: 0x0c0c12, roughness: 0.08 });
const SHINE = new THREE.MeshBasicMaterial({ color: 0xffffff });
const furs = new Map<number, THREE.MeshPhysicalMaterial>();

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

type Part = readonly [material: THREE.Material, sx: number, sy: number, sz: number, x: number, y: number, z: number];

function ears(look: PlushLook): readonly Part[] {
  const trim = fur(look.trim);
  const inner = fur(look.accent);
  switch (look.ears) {
    case "round":
      return [-1, 1].flatMap((side): Part[] => [
        [trim, 0.3, 0.3, 0.16, side * 0.62, 0.78, 0],
        [inner, 0.17, 0.17, 0.08, side * 0.62, 0.78, 0.1],
      ]);
    case "long":
      return [-1, 1].flatMap((side): Part[] => [
        [trim, 0.17, 0.5, 0.11, side * 0.34, 1.2, 0],
        [inner, 0.09, 0.36, 0.05, side * 0.34, 1.2, 0.08],
      ]);
    case "pointy":
    case "none":
      return [];
    default: {
      const unreachable: never = look.ears;
      return unreachable;
    }
  }
}

export function createPlush(lookIndex: number, radius: number): THREE.Group {
  const look = PLUSH_LOOKS[lookIndex % PLUSH_LOOKS.length];
  if (!look) throw new Error(`unknown plush look ${lookIndex}`);
  const coat = fur(look.fur);
  const trim = fur(look.trim);
  const accent = fur(look.accent);
  const parts: readonly Part[] = [
    [coat, 1, 0.97, 1, 0, 0, 0],
    [accent, 0.5, 0.42, 0.2, 0, -0.3, 0.8],
    [accent, 0.26, 0.19, 0.16, 0, 0.1, 0.9],
    [EYE, 0.06, 0.045, 0.04, 0, 0.19, 1.04],
    [EYE, 0.1, 0.11, 0.07, -0.34, 0.34, 0.86],
    [EYE, 0.1, 0.11, 0.07, 0.34, 0.34, 0.86],
    [SHINE, 0.03, 0.03, 0.02, -0.31, 0.39, 0.92],
    [SHINE, 0.03, 0.03, 0.02, 0.37, 0.39, 0.92],
    [trim, 0.24, 0.2, 0.3, -0.45, -0.86, 0.38],
    [trim, 0.24, 0.2, 0.3, 0.45, -0.86, 0.38],
    [trim, 0.2, 0.28, 0.2, -0.93, -0.12, 0.3],
    [trim, 0.2, 0.28, 0.2, 0.93, -0.12, 0.3],
    ...ears(look),
  ];
  const group = new THREE.Group();
  for (const [material, sx, sy, sz, x, y, z] of parts) {
    const part = new THREE.Mesh(BALL, material);
    part.scale.set(sx * radius, sy * radius, sz * radius);
    part.position.set(x * radius, y * radius, z * radius);
    part.castShadow = true;
    part.receiveShadow = true;
    group.add(part);
  }
  if (look.ears === "pointy") {
    for (const side of [-1, 1]) {
      const ear = new THREE.Mesh(CONE, trim);
      ear.scale.set(0.26 * radius, 0.42 * radius, 0.16 * radius);
      ear.position.set(side * 0.52 * radius, 0.98 * radius, 0);
      ear.rotation.z = -side * 0.35;
      ear.castShadow = true;
      group.add(ear);
    }
  }
  return group;
}
