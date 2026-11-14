import { CylinderGeometry, Group, Mesh, MeshStandardMaterial, SphereGeometry } from "three";
import { seeded } from "../art/canvas";
import type { Platform } from "../scenes/stations";
import { PAGE_D, PAGE_W } from "./popup";

function cloudBed(): Group {
  const rand = seeded(9);
  const g = new Group();
  const white = new MeshStandardMaterial({ color: "#ffffff", roughness: 1 });
  for (let i = 0; i < 26; i++) {
    const r = 0.9 + rand() * 1.1;
    const puff = new Mesh(new SphereGeometry(r, 16, 12), white);
    const a = (i / 26) * Math.PI * 2;
    puff.position.set(Math.cos(a) * (PAGE_W + 0.2) * (0.7 + rand() * 0.3), -0.6 - rand() * 0.8, Math.sin(a) * (PAGE_D / 2 + 0.4) * (0.7 + rand() * 0.3));
    puff.scale.y = 0.6;
    g.add(puff);
  }
  return g;
}

function giantLeaf(): Group {
  const green = new MeshStandardMaterial({ color: "#4fbf3e", roughness: 0.55 });
  const leaf = new Mesh(new SphereGeometry(1, 32, 16), green);
  leaf.scale.set(PAGE_W * 1.35, 0.25, PAGE_D * 0.75);
  leaf.position.y = -0.1;
  leaf.receiveShadow = true;
  const stem = new Mesh(new CylinderGeometry(0.3, 0.45, 4.2, 12), new MeshStandardMaterial({ color: "#3fa83a", roughness: 0.6 }));
  stem.rotation.x = Math.PI / 2;
  stem.position.set(0, -0.15, -PAGE_D * 0.75 - 1.6);
  const g = new Group();
  g.add(leaf, stem);
  return g;
}

export function platform(kind: Platform): Group {
  const g = new Group();
  if (kind === "cloud") g.add(cloudBed());
  if (kind === "leaf") g.add(giantLeaf());
  return g;
}
