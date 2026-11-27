import * as THREE from "three";
import { PALETTE } from "./palette";
import {
  createFacadeMaterials,
  createGableRoof,
  createOutlinedBox,
  createOutlinedCylinder,
} from "./geometry";
import { createFacadeTexture } from "./textures";

type VictorianSpec = {
  readonly x: number;
  readonly z: number;
  readonly bodyHex: string;
  readonly roofColor: number;
  readonly rotateY: number;
  readonly height?: number;
};

type ShopSpec = {
  readonly x: number;
  readonly z: number;
  readonly width: number;
  readonly height: number;
  readonly depth: number;
  readonly baseHex: string;
  readonly roofColor: number;
  readonly awningColor?: number;
  readonly waterTower?: boolean;
};

type TowerSpec = {
  readonly x: number;
  readonly z: number;
  readonly width: number;
  readonly height: number;
  readonly depth: number;
  readonly baseHex: string;
  readonly windowHex: string;
  readonly roofColor: number;
  readonly cols: number;
  readonly rows: number;
  readonly penthouse?: boolean;
};

function buildVictorian(spec: VictorianSpec, tagPaintedLadies: boolean): THREE.Group {
  const house = new THREE.Group();
  const h = spec.height ?? 1.95;

  const facadeTex = createFacadeTexture({
    base: spec.bodyHex,
    windowColor: "#8eb4c9",
    cols: 2,
    rows: 2,
    stripeColor: "#2b2320",
  });
  const body = createOutlinedBox({
    width: 1.45,
    height: h,
    depth: 1.15,
    material: createFacadeMaterials(facadeTex, spec.roofColor),
  });
  body.position.set(0, h / 2, 0);
  house.add(body);

  const bay = createOutlinedBox({
    width: 0.34,
    height: h * 0.72,
    depth: 0.62,
    color: spec.roofColor,
  });
  bay.position.set(0.74, h * 0.42, 0);
  house.add(bay);

  const roof = createGableRoof(1.22, 0.82, 1.56, spec.roofColor);
  roof.rotation.y = Math.PI / 2;
  roof.position.set(0, h, 0);
  house.add(roof);

  const chimney = createOutlinedBox({
    width: 0.22,
    height: 0.6,
    depth: 0.22,
    color: PALETTE.warmBrick,
  });
  chimney.position.set(-0.32, h + 0.52, -0.32);
  house.add(chimney);

  const stoop = createOutlinedBox({
    width: 0.38,
    height: 0.28,
    depth: 0.44,
    color: PALETTE.sidewalk,
  });
  stoop.position.set(0.86, 0.14, 0.34);
  house.add(stoop);

  house.position.set(spec.x, 0.04, spec.z);
  house.rotation.y = spec.rotateY;
  if (tagPaintedLadies) {
    house.traverse((obj) => {
      obj.userData.landmarkId = "painted-ladies";
    });
  }
  return house;
}

function buildShopBlock(spec: ShopSpec): THREE.Group {
  const block = new THREE.Group();
  const cols = Math.max(2, Math.round(spec.width * 1.4));
  const rows = Math.max(2, Math.round(spec.height * 1.1));
  const tex = createFacadeTexture({
    base: spec.baseHex,
    windowColor: "#7ea8be",
    cols,
    rows,
  });

  const body = createOutlinedBox({
    width: spec.width,
    height: spec.height,
    depth: spec.depth,
    material: createFacadeMaterials(tex, spec.roofColor),
  });
  body.position.set(0, spec.height / 2, 0);
  block.add(body);

  const parapet = createOutlinedBox({
    width: spec.width + 0.1,
    height: 0.14,
    depth: spec.depth + 0.1,
    color: spec.roofColor,
  });
  parapet.position.set(0, spec.height + 0.06, 0);
  block.add(parapet);

  const ac = createOutlinedBox({
    width: 0.46,
    height: 0.26,
    depth: 0.38,
    color: PALETTE.towerSteel,
  });
  ac.position.set(-spec.width * 0.22, spec.height + 0.22, -spec.depth * 0.14);
  block.add(ac);

  if (spec.awningColor) {
    const awning = createOutlinedBox({
      width: spec.width * 0.72,
      height: 0.16,
      depth: 0.42,
      color: spec.awningColor,
    });
    awning.position.set(0, 0.72, spec.depth / 2 + 0.2);
    awning.rotation.x = 0.26;
    block.add(awning);
  }

  if (spec.waterTower) {
    const tank = createOutlinedCylinder(0.26, 0.28, 0.56, 8, PALETTE.pierWood);
    tank.position.set(spec.width * 0.22, spec.height + 0.58, spec.depth * 0.15);
    block.add(tank);

    const lid = createOutlinedCylinder(0.04, 0.3, 0.22, 8, PALETTE.trunk);
    lid.position.set(spec.width * 0.22, spec.height + 0.96, spec.depth * 0.15);
    block.add(lid);
  }

  block.position.set(spec.x, 0.04, spec.z);
  return block;
}

function buildSkyscraper(spec: TowerSpec): THREE.Group {
  const tower = new THREE.Group();
  const tex = createFacadeTexture({
    base: spec.baseHex,
    windowColor: spec.windowHex,
    cols: spec.cols,
    rows: spec.rows,
  });

  const body = createOutlinedBox({
    width: spec.width,
    height: spec.height,
    depth: spec.depth,
    material: createFacadeMaterials(tex, spec.roofColor),
  });
  body.position.set(0, spec.height / 2, 0);
  tower.add(body);

  const crown = createOutlinedBox({
    width: spec.width + 0.12,
    height: 0.18,
    depth: spec.depth + 0.12,
    color: spec.roofColor,
  });
  crown.position.set(0, spec.height + 0.09, 0);
  tower.add(crown);

  if (spec.penthouse) {
    const mech = createOutlinedBox({
      width: spec.width * 0.56,
      height: 0.65,
      depth: spec.depth * 0.56,
      color: PALETTE.sidewalk,
    });
    mech.position.set(0, spec.height + 0.45, 0);
    tower.add(mech);

    const antenna = createOutlinedCylinder(0.04, 0.06, 1.0, 6, PALETTE.ink);
    antenna.position.set(0.32, spec.height + 1.15, -0.2);
    tower.add(antenna);
  }

  tower.position.set(spec.x, 0.04, spec.z);
  return tower;
}

export function buildCityBlocks(): THREE.Group {
  const group = new THREE.Group();

  const paintedLadies: readonly VictorianSpec[] = [
    { x: -9.5, z: 9.7, bodyHex: "#c69e74", roofColor: PALETTE.victorianBrown, rotateY: 0, height: 2.0 },
    { x: -9.5, z: 8.2, bodyHex: "#b3815a", roofColor: PALETTE.victorianTan, rotateY: 0, height: 2.1 },
    { x: -9.5, z: 6.7, bodyHex: "#8094a6", roofColor: PALETTE.victorianSlate, rotateY: 0, height: 2.0 },
    { x: -9.5, z: 5.2, bodyHex: "#bb5748", roofColor: PALETTE.terracotta, rotateY: 0, height: 1.9 },
    { x: -9.5, z: 3.7, bodyHex: "#d7a855", roofColor: PALETTE.victorianBrown, rotateY: 0, height: 2.05 },
    { x: -9.5, z: 2.2, bodyHex: "#6285a8", roofColor: PALETTE.ferryBlue, rotateY: 0, height: 1.95 },
  ];

  for (const spec of paintedLadies) {
    group.add(buildVictorian(spec, true));
  }

  const frontVictorians: readonly VictorianSpec[] = [
    { x: -1.1, z: 7.5, bodyHex: "#9b7584", roofColor: PALETTE.victorianMauve, rotateY: -Math.PI / 2, height: 1.95 },
    { x: 0.45, z: 7.5, bodyHex: "#c66d85", roofColor: PALETTE.victorianPink, rotateY: -Math.PI / 2, height: 1.85 },
  ];
  for (const spec of frontVictorians) {
    group.add(buildVictorian(spec, false));
  }

  const shops: readonly ShopSpec[] = [
    { x: 2.5, z: 7.45, width: 2.1, height: 2.35, depth: 1.9, baseHex: "#d2be9e", roofColor: PALETTE.sidewalk, awningColor: PALETTE.awningGreen },
    { x: 4.75, z: 7.45, width: 2.05, height: 2.85, depth: 1.9, baseHex: "#b86e54", roofColor: PALETTE.terracotta, awningColor: PALETTE.awningBlue },
    { x: 6.9, z: 7.45, width: 1.95, height: 3.45, depth: 1.9, baseHex: "#9f5946", roofColor: PALETTE.warmBrick, awningColor: PALETTE.awningBlue, waterTower: true },
    { x: 0.35, z: 2.55, width: 1.9, height: 2.2, depth: 1.75, baseHex: "#c8d0d6", roofColor: PALETTE.towerSteel, awningColor: PALETTE.awningBlue },
    { x: 2.45, z: 2.55, width: 2.0, height: 2.65, depth: 1.75, baseHex: "#ccb391", roofColor: PALETTE.sandstone, awningColor: PALETTE.awningGreen, waterTower: true },
    { x: 4.55, z: 2.55, width: 1.85, height: 3.1, depth: 1.75, baseHex: "#b56d53", roofColor: PALETTE.terracotta },
    { x: 4.35, z: -0.25, width: 1.9, height: 3.6, depth: 1.9, baseHex: "#b87458", roofColor: PALETTE.warmBrick, waterTower: true },
  ];
  for (const spec of shops) {
    group.add(buildShopBlock(spec));
  }

  const towers: readonly TowerSpec[] = [
    { x: 6.9, z: -8.6, width: 2.55, height: 6.8, depth: 2.45, baseHex: "#dfd9ce", windowHex: "#5f7484", roofColor: PALETTE.towerWhite, cols: 5, rows: 9, penthouse: true },
    { x: 5.5, z: -5.3, width: 2.15, height: 5.1, depth: 2.15, baseHex: "#8da0ae", windowHex: "#495d6d", roofColor: PALETTE.towerSteel, cols: 4, rows: 7 },
    { x: 9.25, z: -6.3, width: 2.4, height: 6.6, depth: 2.35, baseHex: "#627482", windowHex: "#8fb7cc", roofColor: PALETTE.towerDark, cols: 5, rows: 10, penthouse: true },
    { x: 9.45, z: -3.2, width: 2.25, height: 4.7, depth: 2.15, baseHex: "#d2c3ab", windowHex: "#637b8e", roofColor: PALETTE.sandstone, cols: 4, rows: 7 },
    { x: 8.6, z: -9.4, width: 2.1, height: 4.9, depth: 2.1, baseHex: "#b9c4cc", windowHex: "#4c5f6e", roofColor: PALETTE.towerWhite, cols: 4, rows: 7 },
  ];
  for (const spec of towers) {
    group.add(buildSkyscraper(spec));
  }

  const phoneBooth = createOutlinedBox({
    width: 0.34,
    height: 0.72,
    depth: 0.34,
    color: PALETTE.phoneRed,
  });
  phoneBooth.position.set(1.35, 0.4, 3.65);
  group.add(phoneBooth);

  const mailboxBlue = createOutlinedBox({
    width: 0.24,
    height: 0.38,
    depth: 0.24,
    color: PALETTE.carBlue,
  });
  mailboxBlue.position.set(4.1, 0.24, 8.6);
  group.add(mailboxBlue);

  const mailboxRed = createOutlinedBox({
    width: 0.24,
    height: 0.38,
    depth: 0.24,
    color: PALETTE.phoneRed,
  });
  mailboxRed.position.set(4.45, 0.24, 8.6);
  group.add(mailboxRed);

  return group;
}
