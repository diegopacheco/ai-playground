import * as THREE from "three";
import { PALETTE } from "./palette";
import { addInkEdges, createOutlinedBox } from "./geometry";
import {
  createGrassTexture,
  createRoadTexture,
  createSoilTexture,
  createWaterTexture,
} from "./textures";

export type TerrainRig = {
  readonly group: THREE.Group;
  readonly waterTexture: THREE.CanvasTexture;
  readonly bayWaterMesh: THREE.Mesh;
};

function addCrosswalk(
  group: THREE.Group,
  x: number,
  z: number,
  horizontal: boolean,
  stripes = 5,
): void {
  const mat = new THREE.MeshLambertMaterial({ color: PALETTE.roadLine });
  const spacing = 0.28;
  const start = -((stripes - 1) * spacing) / 2;
  for (let i = 0; i < stripes; i++) {
    const offset = start + i * spacing;
    const w = horizontal ? 0.56 : 0.14;
    const d = horizontal ? 0.14 : 0.56;
    const bar = new THREE.Mesh(new THREE.BoxGeometry(w, 0.03, d), mat);
    bar.position.set(
      horizontal ? x : x + offset,
      0.045,
      horizontal ? z + offset : z,
    );
    bar.receiveShadow = true;
    group.add(bar);
  }
}

function addRoadSegment(
  group: THREE.Group,
  x: number,
  z: number,
  width: number,
  depth: number,
  vertical: boolean,
  dashed = true,
): void {
  const sidewalk = createOutlinedBox({
    width: width + 0.48,
    height: 0.06,
    depth: depth + 0.48,
    color: PALETTE.sidewalk,
    castShadow: false,
    receiveShadow: true,
    outline: true,
  });
  sidewalk.position.set(x, 0.01, z);
  group.add(sidewalk);

  const repX = vertical ? 1 : Math.max(1, Math.round(width / 2.2));
  const repY = vertical ? Math.max(1, Math.round(depth / 2.2)) : 1;
  const roadTex = createRoadTexture(dashed, vertical, repX, repY);
  const roadMat = new THREE.MeshLambertMaterial({ map: roadTex });
  const road = new THREE.Mesh(new THREE.BoxGeometry(width, 0.04, depth), roadMat);
  road.position.set(x, 0.035, z);
  road.receiveShadow = true;
  group.add(road);
}

export function buildTerrain(): TerrainRig {
  const group = new THREE.Group();
  const size = 24;
  const soilHeight = 2.4;

  const soilTex = createSoilTexture(6, 1);
  const grassTex = createGrassTexture(8, 8);
  const soilSideMat = new THREE.MeshLambertMaterial({ map: soilTex });
  const grassTopMat = new THREE.MeshLambertMaterial({ map: grassTex });
  const bottomMat = new THREE.MeshLambertMaterial({ color: PALETTE.soilDeep });

  const slabGeo = new THREE.BoxGeometry(size, soilHeight, size);
  const slab = new THREE.Mesh(slabGeo, [
    soilSideMat,
    soilSideMat,
    grassTopMat,
    bottomMat,
    soilSideMat,
    soilSideMat,
  ]);
  slab.position.set(0, -soilHeight / 2, 0);
  slab.receiveShadow = true;
  slab.castShadow = true;
  addInkEdges(slab, 25);
  group.add(slab);

  const waterTexture = createWaterTexture(4, 3);
  const waterMat = new THREE.MeshLambertMaterial({ map: waterTexture });

  const bayWaterGeo = new THREE.BoxGeometry(10.6, 0.14, 7.6);
  const bayWaterMesh = new THREE.Mesh(bayWaterGeo, waterMat);
  bayWaterMesh.position.set(-5.8, -0.04, -7.8);
  bayWaterMesh.receiveShadow = true;
  addInkEdges(bayWaterMesh, 30);
  group.add(bayWaterMesh);

  const bayWall = createOutlinedBox({
    width: 10.9,
    height: 0.24,
    depth: 0.34,
    color: PALETTE.sidewalk,
    castShadow: false,
  });
  bayWall.position.set(-5.6, 0.02, -3.9);
  group.add(bayWall);

  const bayRightWall = createOutlinedBox({
    width: 0.34,
    height: 0.24,
    depth: 7.9,
    color: PALETTE.sidewalk,
    castShadow: false,
  });
  bayRightWall.position.set(-0.4, 0.02, -7.8);
  group.add(bayRightWall);

  const headland = createOutlinedBox({
    width: 2.6,
    height: 0.45,
    depth: 2.4,
    color: PALETTE.grassLight,
    castShadow: true,
  });
  headland.position.set(-10.5, 0.15, -10.5);
  group.add(headland);

  const pierDeck = createOutlinedBox({
    width: 2.2,
    height: 0.12,
    depth: 0.95,
    color: PALETTE.pierWood,
    castShadow: true,
  });
  pierDeck.position.set(-1.55, 0.12, -5.6);
  group.add(pierDeck);

  for (const px of [-2.4, -1.7, -1.0]) {
    for (const pz of [-5.95, -5.25]) {
      const post = createOutlinedBox({
        width: 0.14,
        height: 0.34,
        depth: 0.14,
        color: PALETTE.trunk,
      });
      post.position.set(px, 0.06, pz);
      group.add(post);
    }
  }

  addRoadSegment(group, -6.8, 6.2, 1.9, 10.8, true, true);
  addRoadSegment(group, 1.4, 4.9, 16.4, 1.85, false, true);
  addRoadSegment(group, 3.8, 9.7, 11.4, 1.85, false, true);
  addRoadSegment(group, 9.5, 2.2, 1.9, 16.4, true, true);
  addRoadSegment(group, 1.2, -5.5, 1.85, 8.6, true, true);
  addRoadSegment(group, 4.4, -2.0, 8.4, 1.7, false, true);

  const trackMat = new THREE.MeshLambertMaterial({ color: PALETTE.ink });
  for (const tz of [4.62, 5.18]) {
    const rail = new THREE.Mesh(new THREE.BoxGeometry(14.8, 0.03, 0.07), trackMat);
    rail.position.set(0.8, 0.06, tz);
    group.add(rail);
  }
  const slot = new THREE.Mesh(new THREE.BoxGeometry(14.8, 0.025, 0.05), trackMat);
  slot.position.set(0.8, 0.06, 4.9);
  group.add(slot);

  addCrosswalk(group, -6.8, 3.7, false);
  addCrosswalk(group, -5.5, 4.9, true);
  addCrosswalk(group, 9.5, 3.6, false);
  addCrosswalk(group, 8.1, 4.9, true);
  addCrosswalk(group, 8.1, 9.7, true);
  addCrosswalk(group, 9.5, 8.3, false);
  addCrosswalk(group, 1.2, -3.2, false);

  const plaza = createOutlinedBox({
    width: 3.1,
    height: 0.07,
    depth: 5.8,
    color: PALETTE.sidewalk,
    castShadow: false,
  });
  plaza.position.set(-0.2, 0.02, -4.7);
  group.add(plaza);

  const parkPath1 = createOutlinedBox({
    width: 3.6,
    height: 0.05,
    depth: 0.45,
    color: PALETTE.sidewalk,
    castShadow: false,
    outline: false,
  });
  parkPath1.position.set(-3.2, 0.02, 0.4);
  group.add(parkPath1);

  const parkPath2 = createOutlinedBox({
    width: 0.45,
    height: 0.05,
    depth: 4.8,
    color: PALETTE.sidewalk,
    castShadow: false,
    outline: false,
  });
  parkPath2.position.set(-3.2, 0.02, 0.5);
  group.add(parkPath2);

  return { group, waterTexture, bayWaterMesh };
}
