import * as THREE from "three";
import type { SurfaceId } from "../types";
import { clayMaterial, glowMaterial } from "./materials";

export type DollhouseSurfaceRefs = {
  readonly group: THREE.Group;
  readonly materials: Readonly<Record<SurfaceId, THREE.MeshStandardMaterial>>;
  readonly clouds: readonly THREE.Group[];
};

function buildWindow(x: number, y: number, z: number): THREE.Group {
  const win = new THREE.Group();
  win.position.set(x, y, z);

  const frameMat = clayMaterial("#ffffff", 0.5);
  const skyMat = glowMaterial("#a5e4ff", 0.35);

  const pane = new THREE.Mesh(new THREE.BoxGeometry(1.15, 0.95, 0.06), skyMat);
  win.add(pane);

  const borderTop = new THREE.Mesh(new THREE.BoxGeometry(1.3, 0.12, 0.12), frameMat);
  borderTop.position.y = 0.5;
  win.add(borderTop);

  const borderBot = new THREE.Mesh(new THREE.BoxGeometry(1.34, 0.14, 0.18), frameMat);
  borderBot.position.y = -0.5;
  win.add(borderBot);

  const barV = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.98, 0.1), frameMat);
  win.add(barV);

  const barH = new THREE.Mesh(new THREE.BoxGeometry(1.18, 0.08, 0.1), frameMat);
  win.add(barH);

  return win;
}

function buildCloud(x: number, y: number, z: number, scale: number): THREE.Group {
  const cloud = new THREE.Group();
  cloud.position.set(x, y, z);
  cloud.scale.setScalar(scale);
  const mat = clayMaterial("#ffffff", 0.9);
  for (const [px, py, r] of [[-0.5, 0, 0.45], [0, 0.18, 0.58], [0.52, 0.02, 0.44]] as const) {
    const puff = new THREE.Mesh(new THREE.SphereGeometry(r, 14, 12), mat);
    puff.position.set(px, py, 0);
    cloud.add(puff);
  }
  return cloud;
}

export function buildDollhouseShell(surfaces: Readonly<Record<SurfaceId, string>>): DollhouseSurfaceRefs {
  const group = new THREE.Group();

  const materials: Record<SurfaceId, THREE.MeshStandardMaterial> = {
    kitchenWall: clayMaterial(surfaces.kitchenWall, 0.8),
    livingWall: clayMaterial(surfaces.livingWall, 0.8),
    bedroomWall: clayMaterial(surfaces.bedroomWall, 0.8),
    floor: clayMaterial(surfaces.floor, 0.72),
    roof: clayMaterial(surfaces.roof, 0.58),
  };

  const grass = new THREE.Mesh(
    new THREE.CylinderGeometry(20, 20, 12, 48),
    clayMaterial("#9ee69a", 0.9),
  );
  grass.position.set(0, -6.15, 0);
  grass.receiveShadow = true;
  group.add(grass);

  const flowerColors = ["#ff8fab", "#ffe066", "#74c0fc", "#cbb2fe"] as const;
  for (let i = 0; i < 10; i += 1) {
    const angle = (i / 10) * Math.PI * 2;
    const dist = 8.4 + (i % 2) * 1.1;
    const flower = new THREE.Mesh(
      new THREE.SphereGeometry(0.22, 10, 8),
      clayMaterial(flowerColors[i % flowerColors.length] ?? "#ffe066", 0.6),
    );
    flower.position.set(Math.cos(angle) * dist, -0.05, Math.sin(angle) * 3.8 + 1.4);
    group.add(flower);
  }

  const floor = new THREE.Mesh(new THREE.BoxGeometry(13.6, 0.3, 4.4), materials.floor);
  floor.position.set(0, -0.15, -0.1);
  floor.receiveShadow = true;
  floor.userData = { hitKind: "surface", hitId: "floor" };
  group.add(floor);

  const roomWalls: readonly { readonly id: SurfaceId; readonly x: number }[] = [
    { id: "kitchenWall", x: -4.4 },
    { id: "livingWall", x: 0 },
    { id: "bedroomWall", x: 4.4 },
  ];

  for (const rw of roomWalls) {
    const wall = new THREE.Mesh(new THREE.BoxGeometry(4.42, 3.6, 0.24), materials[rw.id]);
    wall.position.set(rw.x, 1.8, -2.18);
    wall.receiveShadow = true;
    wall.userData = { hitKind: "surface", hitId: rw.id };
    group.add(wall);
  }

  const trimMat = clayMaterial("#fff7e8", 0.62);
  for (const sx of [-6.7, -2.2, 2.2, 6.7]) {
    const divider = new THREE.Mesh(
      new THREE.BoxGeometry(0.2, 3.6, sx === -6.7 || sx === 6.7 ? 4.4 : 2.2),
      trimMat,
    );
    divider.position.set(sx, 1.8, sx === -6.7 || sx === 6.7 ? -0.1 : -1.0);
    divider.castShadow = true;
    divider.receiveShadow = true;
    group.add(divider);
  }

  const ceiling = new THREE.Mesh(new THREE.BoxGeometry(13.8, 0.24, 4.5), trimMat);
  ceiling.position.set(0, 3.7, -0.1);
  ceiling.castShadow = true;
  group.add(ceiling);

  const roofShape = new THREE.Shape();
  roofShape.moveTo(-7.3, 0);
  roofShape.lineTo(0, 2.55);
  roofShape.lineTo(7.3, 0);
  roofShape.closePath();

  const roofGeo = new THREE.ExtrudeGeometry(roofShape, {
    depth: 4.7,
    bevelEnabled: true,
    bevelThickness: 0.14,
    bevelSize: 0.14,
    bevelSegments: 3,
  });
  const roof = new THREE.Mesh(roofGeo, materials.roof);
  roof.position.set(0, 3.75, -2.45);
  roof.castShadow = true;
  roof.userData = { hitKind: "surface", hitId: "roof" };
  group.add(roof);

  const chimney = new THREE.Mesh(new THREE.BoxGeometry(0.9, 1.4, 0.9), clayMaterial("#ffbe76", 0.65));
  chimney.position.set(3.8, 5.1, -0.8);
  chimney.castShadow = true;
  group.add(chimney);

  const atticWindow = new THREE.Mesh(new THREE.CylinderGeometry(0.52, 0.52, 0.18, 22), glowMaterial("#fff3b0", 0.55));
  atticWindow.rotation.x = Math.PI / 2;
  atticWindow.position.set(0, 4.85, 2.34);
  group.add(atticWindow);

  const atticFrame = new THREE.Mesh(new THREE.TorusGeometry(0.54, 0.08, 10, 24), trimMat);
  atticFrame.position.set(0, 4.85, 2.42);
  group.add(atticFrame);

  group.add(buildWindow(-4.4, 2.25, -2.02));
  group.add(buildWindow(0.0, 2.25, -2.02));
  group.add(buildWindow(4.4, 2.25, -2.02));

  const clouds = [
    buildCloud(-8.5, 6.2, -5.5, 1.15),
    buildCloud(0.5, 7.2, -6.5, 1.3),
    buildCloud(8.8, 6.0, -5.0, 1.05),
  ] as const;
  for (const c of clouds) {
    group.add(c);
  }

  return { group, materials, clouds };
}
