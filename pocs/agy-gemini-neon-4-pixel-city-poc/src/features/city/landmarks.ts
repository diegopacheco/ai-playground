import * as THREE from "three";
import { PALETTE } from "./palette";
import {
  addInkEdges,
  createFacadeMaterials,
  createOutlinedBox,
  createOutlinedCylinder,
} from "./geometry";
import {
  createFacadeTexture,
  createGlassTowerTexture,
  createPyramidTexture,
} from "./textures";
import type { LandmarkId } from "./types";

export type LandmarkRig = {
  readonly group: THREE.Group;
  readonly windowMaterials: THREE.MeshLambertMaterial[];
};

function tagLandmark(root: THREE.Object3D, id: LandmarkId): void {
  root.traverse((obj) => {
    obj.userData.landmarkId = id;
  });
}

function buildGoldenGateBridge(): THREE.Group {
  const bridge = new THREE.Group();
  const centerX = -6.8;
  const deckY = 1.15;
  const spanStartZ = 1.6;
  const spanEndZ = -10.8;
  const spanLen = spanStartZ - spanEndZ;
  const midZ = (spanStartZ + spanEndZ) / 2;

  const deck = createOutlinedBox({
    width: 1.95,
    height: 0.22,
    depth: spanLen,
    color: PALETTE.roadAsphalt,
  });
  deck.position.set(centerX, deckY, midZ);
  bridge.add(deck);

  for (const side of [-0.95, 0.95]) {
    const girder = createOutlinedBox({
      width: 0.14,
      height: 0.36,
      depth: spanLen,
      color: PALETTE.bridgeRed,
    });
    girder.position.set(centerX + side, deckY + 0.08, midZ);
    bridge.add(girder);
  }

  const southRamp = createOutlinedBox({
    width: 1.95,
    height: 0.24,
    depth: 2.0,
    color: PALETTE.roadAsphalt,
  });
  southRamp.position.set(centerX, deckY / 2, spanStartZ + 0.75);
  southRamp.rotation.x = 0.54;
  bridge.add(southRamp);

  const towerZs: readonly number[] = [-1.8, -6.9];
  const towerHeight = 5.4;

  for (const tz of towerZs) {
    const concreteBase = createOutlinedBox({
      width: 2.8,
      height: 0.85,
      depth: 1.0,
      color: PALETTE.sidewalk,
    });
    concreteBase.position.set(centerX, 0.28, tz);
    bridge.add(concreteBase);

    for (const side of [-0.96, 0.96]) {
      const legLower = createOutlinedBox({
        width: 0.38,
        height: towerHeight * 0.52,
        depth: 0.42,
        color: PALETTE.bridgeRed,
      });
      legLower.position.set(centerX + side, 0.55 + towerHeight * 0.26, tz);
      bridge.add(legLower);

      const legUpper = createOutlinedBox({
        width: 0.3,
        height: towerHeight * 0.48,
        depth: 0.34,
        color: PALETTE.bridgeRed,
      });
      legUpper.position.set(centerX + side, 0.55 + towerHeight * 0.74, tz);
      bridge.add(legUpper);

      const cap = createOutlinedBox({
        width: 0.42,
        height: 0.18,
        depth: 0.44,
        color: PALETTE.bridgeRedDark,
      });
      cap.position.set(centerX + side, 0.55 + towerHeight + 0.06, tz);
      bridge.add(cap);
    }

    for (const barY of [2.55, 3.65, 4.75, 5.75]) {
      const cross = createOutlinedBox({
        width: 1.9,
        height: 0.28,
        depth: 0.24,
        color: PALETTE.bridgeRed,
      });
      cross.position.set(centerX, barY, tz);
      bridge.add(cross);
    }
  }

  const cableMat = new THREE.MeshLambertMaterial({ color: PALETTE.bridgeCable });
  const suspenderMat = new THREE.MeshLambertMaterial({ color: PALETTE.bridgeRed });

  const towerTopY = 5.9;
  const sagMidY = 2.05;
  const anchorY = deckY + 0.28;

  for (const side of [-0.96, 0.96]) {
    const x = centerX + side;
    const southCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(x, anchorY, spanStartZ),
      new THREE.Vector3(x, (anchorY + towerTopY) * 0.45, 0.1),
      new THREE.Vector3(x, towerTopY, -1.8),
    ]);
    const mainCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(x, towerTopY, -1.8),
      new THREE.Vector3(x, sagMidY + 0.65, -3.1),
      new THREE.Vector3(x, sagMidY, -4.35),
      new THREE.Vector3(x, sagMidY + 0.65, -5.6),
      new THREE.Vector3(x, towerTopY, -6.9),
    ]);
    const northCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(x, towerTopY, -6.9),
      new THREE.Vector3(x, (anchorY + towerTopY) * 0.45, -8.8),
      new THREE.Vector3(x, anchorY, spanEndZ),
    ]);

    for (const curve of [southCurve, mainCurve, northCurve]) {
      const tubeGeo = new THREE.TubeGeometry(curve, 18, 0.065, 6, false);
      const tube = new THREE.Mesh(tubeGeo, cableMat);
      tube.castShadow = true;
      bridge.add(tube);

      const samples = curve.getPoints(7);
      for (let i = 1; i < samples.length - 1; i++) {
        const pt = samples[i];
        if (!pt) {
          continue;
        }
        const h = pt.y - (deckY + 0.22);
        if (h > 0.25) {
          const cable = new THREE.Mesh(
            new THREE.BoxGeometry(0.04, h, 0.04),
            suspenderMat,
          );
          cable.position.set(x, deckY + 0.22 + h / 2, pt.z);
          bridge.add(cable);
        }
      }
    }
  }

  tagLandmark(bridge, "golden-gate");
  return bridge;
}

function buildFerryBuilding(): THREE.Group {
  const ferry = new THREE.Group();
  const baseX = -0.2;
  const baseZ = -5.6;

  const hallTex = createFacadeTexture({
    base: "#7292b0",
    windowColor: "#3f5a75",
    frameColor: "#d9e2ea",
    cols: 8,
    rows: 2,
  });
  const hall = createOutlinedBox({
    width: 1.75,
    height: 1.55,
    depth: 4.9,
    material: createFacadeMaterials(hallTex, PALETTE.ferryBlue),
  });
  hall.position.set(baseX, 0.8, baseZ);
  ferry.add(hall);

  const roof = createOutlinedBox({
    width: 1.45,
    height: 0.26,
    depth: 4.5,
    color: PALETTE.ferryTrim,
  });
  roof.position.set(baseX, 1.7, baseZ);
  ferry.add(roof);

  const towerLower = createOutlinedBox({
    width: 1.15,
    height: 2.35,
    depth: 1.15,
    color: PALETTE.ferryBlue,
  });
  towerLower.position.set(baseX, 2.7, baseZ);
  ferry.add(towerLower);

  const clockBox = createOutlinedBox({
    width: 1.22,
    height: 0.95,
    depth: 1.22,
    color: PALETTE.ferryTrim,
  });
  clockBox.position.set(baseX, 4.15, baseZ);
  ferry.add(clockBox);

  const dialMat = new THREE.MeshLambertMaterial({ color: PALETTE.ferryClock });
  const handMat = new THREE.MeshLambertMaterial({ color: PALETTE.ink });
  const dialOffsets: readonly (readonly [number, number, number])[] = [
    [0.62, 0, 0],
    [-0.62, 0, 0],
    [0, 0, 0.62],
    [0, 0, -0.62],
  ];
  for (const [dx, dy, dz] of dialOffsets) {
    const face = new THREE.Mesh(
      new THREE.CylinderGeometry(0.32, 0.32, 0.04, 12),
      dialMat,
    );
    face.position.set(baseX + dx, 4.15 + dy, baseZ + dz);
    if (dx !== 0) {
      face.rotation.z = Math.PI / 2;
    } else {
      face.rotation.x = Math.PI / 2;
    }
    ferry.add(face);

    const pin = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.22, 0.08), handMat);
    pin.position.set(baseX + dx * 1.03, 4.2 + dy, baseZ + dz * 1.03);
    ferry.add(pin);
  }

  const belfry = createOutlinedBox({
    width: 0.72,
    height: 0.72,
    depth: 0.72,
    color: PALETTE.ferryBlue,
  });
  belfry.position.set(baseX, 4.95, baseZ);
  ferry.add(belfry);

  const cupola = createOutlinedCylinder(0.05, 0.36, 0.85, 8, PALETTE.ferryTrim);
  cupola.position.set(baseX, 5.7, baseZ);
  ferry.add(cupola);

  tagLandmark(ferry, "ferry-building");
  return ferry;
}

function buildCoitTower(): THREE.Group {
  const coit = new THREE.Group();
  const cx = -2.4;
  const cz = -1.3;

  const plinth = createOutlinedBox({
    width: 1.55,
    height: 0.45,
    depth: 1.55,
    color: PALETTE.coitCream,
  });
  plinth.position.set(cx, 0.25, cz);
  coit.add(plinth);

  const shaft = createOutlinedCylinder(0.46, 0.56, 2.75, 12, PALETTE.coitCream);
  shaft.position.set(cx, 1.82, cz);
  coit.add(shaft);

  const crown = createOutlinedCylinder(0.52, 0.52, 0.34, 12, PALETTE.sidewalk);
  crown.position.set(cx, 3.25, cz);
  coit.add(crown);

  const lantern = createOutlinedCylinder(0.32, 0.38, 0.42, 10, PALETTE.coitCream);
  lantern.position.set(cx, 3.6, cz);
  coit.add(lantern);

  tagLandmark(coit, "coit-tower");
  return coit;
}

function buildPalaceOfFineArts(): THREE.Group {
  const palace = new THREE.Group();
  const px = -5.4;
  const pz = 8.4;

  const lagoonGeo = new THREE.CylinderGeometry(1.7, 1.75, 0.14, 14);
  const lagoonMat = new THREE.MeshLambertMaterial({ color: PALETTE.lagoonWater });
  const lagoon = new THREE.Mesh(lagoonGeo, lagoonMat);
  lagoon.scale.set(1.25, 1, 0.82);
  lagoon.position.set(px + 0.65, 0.04, pz + 1.05);
  lagoon.receiveShadow = true;
  addInkEdges(lagoon, 35);
  palace.add(lagoon);

  const rimOffsets: readonly (readonly [number, number])[] = [
    [-0.9, 1.4],
    [0.2, 2.2],
    [1.6, 2.1],
    [2.3, 1.2],
    [-0.4, 0.1],
  ];
  for (const [rx, rz] of rimOffsets) {
    const stone = createOutlinedBox({
      width: 0.42,
      height: 0.18,
      depth: 0.34,
      color: PALETTE.sidewalk,
    });
    stone.position.set(px + rx, 0.12, pz + rz);
    palace.add(stone);
  }

  const base = createOutlinedCylinder(1.25, 1.38, 0.3, 12, PALETTE.palaceStone);
  base.position.set(px - 0.15, 0.18, pz - 0.35);
  palace.add(base);

  const columnCount = 8;
  for (let i = 0; i < columnCount; i++) {
    const a = (i / columnCount) * Math.PI * 2;
    const col = createOutlinedCylinder(0.11, 0.12, 1.28, 6, PALETTE.palaceStone);
    col.position.set(
      px - 0.15 + Math.cos(a) * 0.92,
      0.95,
      pz - 0.35 + Math.sin(a) * 0.92,
    );
    palace.add(col);
  }

  const frieze = createOutlinedCylinder(1.18, 1.18, 0.28, 12, PALETTE.palaceStone);
  frieze.position.set(px - 0.15, 1.7, pz - 0.35);
  palace.add(frieze);

  const domeGeo = new THREE.SphereGeometry(
    1.05,
    12,
    8,
    0,
    Math.PI * 2,
    0,
    Math.PI * 0.52,
  );
  const domeMat = new THREE.MeshLambertMaterial({ color: PALETTE.palaceRoof });
  const dome = new THREE.Mesh(domeGeo, domeMat);
  dome.position.set(px - 0.15, 1.78, pz - 0.35);
  dome.castShadow = true;
  addInkEdges(dome, 30);
  palace.add(dome);

  for (let i = 0; i < 4; i++) {
    const angle = -2.6 + i * 0.46;
    const cx = px - 0.15 + Math.cos(angle) * 1.95;
    const cz = pz - 0.35 + Math.sin(angle) * 1.95;
    const pillar = createOutlinedBox({
      width: 0.26,
      height: 1.25,
      depth: 0.26,
      color: PALETTE.palaceStone,
    });
    pillar.position.set(cx, 0.65, cz);
    palace.add(pillar);
  }

  const archBeam = createOutlinedBox({
    width: 2.2,
    height: 0.24,
    depth: 0.4,
    color: PALETTE.palaceRoof,
  });
  archBeam.position.set(px - 1.4, 1.35, pz - 1.45);
  archBeam.rotation.y = Math.PI / 4;
  palace.add(archBeam);

  tagLandmark(palace, "palace-of-fine-arts");
  return palace;
}

function buildTransamericaPyramid(
  windowMaterials: THREE.MeshLambertMaterial[],
): THREE.Group {
  const pyramid = new THREE.Group();
  const tx = 2.6;
  const tz = -8.3;

  const basePodium = createOutlinedBox({
    width: 2.75,
    height: 1.2,
    depth: 2.75,
    color: PALETTE.pyramidStone,
  });
  basePodium.position.set(tx, 0.6, tz);
  pyramid.add(basePodium);

  const pyramidTex = createPyramidTexture(0);
  const pyramidMat = new THREE.MeshLambertMaterial({ map: pyramidTex });
  windowMaterials.push(pyramidMat);

  const shaftGeo = new THREE.ConeGeometry(1.75, 9.0, 4);
  shaftGeo.rotateY(Math.PI / 4);
  const shaft = new THREE.Mesh(shaftGeo, pyramidMat);
  shaft.position.set(tx, 5.6, tz);
  shaft.castShadow = true;
  shaft.receiveShadow = true;
  addInkEdges(shaft, 20);
  pyramid.add(shaft);

  for (const side of [-1, 1]) {
    const wing = createOutlinedBox({
      width: 0.44,
      height: 4.8,
      depth: 0.56,
      color: PALETTE.pyramidStone,
    });
    wing.position.set(tx + side * 0.94, 4.0, tz);
    pyramid.add(wing);
  }

  const spire = createOutlinedCylinder(0.04, 0.22, 1.85, 4, PALETTE.salesforceTrim);
  spire.rotation.y = Math.PI / 4;
  spire.position.set(tx, 10.8, tz);
  pyramid.add(spire);

  tagLandmark(pyramid, "transamerica-pyramid");
  return pyramid;
}

function buildSalesforceTower(
  windowMaterials: THREE.MeshLambertMaterial[],
): THREE.Group {
  const tower = new THREE.Group();
  const sx = 6.8;
  const sz = -0.3;

  const podium = createOutlinedBox({
    width: 2.52,
    height: 1.25,
    depth: 2.52,
    color: PALETTE.salesforceTrim,
  });
  podium.position.set(sx, 0.65, sz);
  tower.add(podium);

  for (const dx of [-0.72, 0, 0.72]) {
    const strut = createOutlinedBox({
      width: 0.16,
      height: 1.22,
      depth: 2.58,
      color: PALETTE.salesforceGlass,
      outline: false,
    });
    strut.position.set(sx + dx, 0.65, sz);
    tower.add(strut);
  }

  const glassTex = createGlassTowerTexture(8, 24, 0);
  const glassMat = new THREE.MeshLambertMaterial({ map: glassTex });
  windowMaterials.push(glassMat);

  const lowerBody = createOutlinedBox({
    width: 2.38,
    height: 5.4,
    depth: 2.38,
    material: [
      glassMat,
      glassMat,
      new THREE.MeshLambertMaterial({ color: PALETTE.salesforceTrim }),
      new THREE.MeshLambertMaterial({ color: PALETTE.ink }),
      glassMat,
      glassMat,
    ],
  });
  lowerBody.position.set(sx, 3.95, sz);
  tower.add(lowerBody);

  const upperGeo = new THREE.CylinderGeometry(1.18, 1.66, 2.75, 4);
  upperGeo.rotateY(Math.PI / 4);
  const upperMesh = new THREE.Mesh(upperGeo, glassMat);
  upperMesh.position.set(sx, 8.02, sz);
  upperMesh.castShadow = true;
  upperMesh.receiveShadow = true;
  addInkEdges(upperMesh, 22);
  tower.add(upperMesh);

  const crown = createOutlinedCylinder(0.86, 1.18, 0.78, 4, PALETTE.salesforceTrim);
  crown.rotation.y = Math.PI / 4;
  crown.position.set(sx, 9.76, sz);
  tower.add(crown);

  tagLandmark(tower, "salesforce-tower");
  return tower;
}

export function buildLandmarks(): LandmarkRig {
  const group = new THREE.Group();
  const windowMaterials: THREE.MeshLambertMaterial[] = [];

  group.add(buildGoldenGateBridge());
  group.add(buildFerryBuilding());
  group.add(buildCoitTower());
  group.add(buildPalaceOfFineArts());
  group.add(buildTransamericaPyramid(windowMaterials));
  group.add(buildSalesforceTower(windowMaterials));

  return { group, windowMaterials };
}
