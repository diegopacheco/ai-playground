import * as THREE from "three";
import "./style.css";

const canvas = document.querySelector("#game");
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: innerWidth > 700,
  alpha: false,
  powerPreference: "high-performance"
});
renderer.setPixelRatio(Math.min(devicePixelRatio, innerWidth < 700 ? 1 : 1.1));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = false;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x91c9d5);
scene.fog = new THREE.Fog(0x91c9d5, 145, 430);

const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.1, 650);
const clock = new THREE.Clock();
const world = new THREE.Group();
scene.add(world);

const colors = {
  ink: 0x1c2b2a,
  paper: 0xf4e8ca,
  orange: 0xef5b3f,
  bridge: 0xe84f37,
  blue: 0x287e9c,
  deepBlue: 0x1c667e,
  green: 0x4d765b,
  darkGreen: 0x315746,
  yellow: 0xf2bc3a,
  red: 0xc94634,
  cream: 0xe9ddb9,
  road: 0x3e4c4a,
  wood: 0x855b3d,
  pink: 0xd7837d
};

const materialCache = new Map();
const mat = (color, options = {}) => {
  const key = `${color}-${options.transparent || false}-${options.opacity || 1}-${options.side ?? THREE.FrontSide}`;
  if (!materialCache.has(key)) {
    materialCache.set(key, new THREE.MeshToonMaterial({
      color,
      transparent: options.transparent,
      opacity: options.opacity ?? 1,
      side: options.side ?? THREE.FrontSide
    }));
  }
  return materialCache.get(key);
};

const inkMaterial = new THREE.LineBasicMaterial({ color: colors.ink, linewidth: 1 });
const geometryCache = {
  box: new THREE.BoxGeometry(1, 1, 1),
  sphere: new THREE.SphereGeometry(0.5, 10, 7),
  cylinder: new THREE.CylinderGeometry(0.5, 0.5, 1, 8),
  cone: new THREE.ConeGeometry(0.5, 1, 8),
  wheel: new THREE.CylinderGeometry(0.5, 0.5, 0.3, 8)
};
const edgeGeometryCache = {
  box: new THREE.EdgesGeometry(geometryCache.box, 28),
  sphere: new THREE.EdgesGeometry(geometryCache.sphere, 28),
  cylinder: new THREE.EdgesGeometry(geometryCache.cylinder, 28),
  cone: new THREE.EdgesGeometry(geometryCache.cone, 28),
  wheel: new THREE.EdgesGeometry(geometryCache.wheel, 28)
};

const roadX = [-145, -80, -15, 50, 115, 180];
const roadZ = [18, 68, 118, 168, 218];
const buildingBounds = [];

function primitive(type, color, position, scale, parent = world, outline = false) {
  const mesh = new THREE.Mesh(geometryCache[type], mat(color));
  mesh.position.set(...position);
  mesh.scale.set(...scale);
  parent.add(mesh);
  if (outline) {
    const edge = new THREE.LineSegments(edgeGeometryCache[type], inkMaterial);
    edge.scale.setScalar(1.003);
    mesh.add(edge);
  }
  return mesh;
}

function box(color, position, scale, parent = world, outline = false) {
  return primitive("box", color, position, scale, parent, outline);
}

function cylinder(color, position, scale, parent = world, outline = false) {
  return primitive("cylinder", color, position, scale, parent, outline);
}

function sphere(color, position, scale, parent = world, outline = false) {
  return primitive("sphere", color, position, scale, parent, outline);
}

function cone(color, position, scale, parent = world, outline = false) {
  return primitive("cone", color, position, scale, parent, outline);
}

function seeded(index) {
  const value = Math.sin(index * 129.73 + 74.21) * 43758.5453;
  return value - Math.floor(value);
}

const hemi = new THREE.HemisphereLight(0xfff3d7, 0x426a67, 2.1);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffefcf, 2.8);
sun.position.set(-90, 150, 40);
scene.add(sun);

const waterGeometry = new THREE.PlaneGeometry(620, 620, 16, 16);
waterGeometry.rotateX(-Math.PI / 2);
const water = new THREE.Mesh(waterGeometry, new THREE.MeshToonMaterial({
  color: colors.blue,
  transparent: true,
  opacity: 0.93
}));
water.position.y = 0;
world.add(water);
const waterPositions = waterGeometry.attributes.position;
for (let index = 0; index < waterPositions.count; index += 1) {
  const x = waterPositions.getX(index);
  const z = waterPositions.getZ(index);
  waterPositions.setY(index, Math.sin(x * 0.07) * 0.28 + Math.cos(z * 0.06) * 0.2);
}
waterPositions.needsUpdate = true;

function createLand() {
  const shape = new THREE.Shape();
  shape.moveTo(-178, -22);
  shape.lineTo(-112, -14);
  shape.lineTo(-72, -28);
  shape.lineTo(-25, -16);
  shape.lineTo(24, -22);
  shape.lineTo(78, -5);
  shape.lineTo(145, -16);
  shape.lineTo(205, 8);
  shape.lineTo(230, 240);
  shape.lineTo(-190, 240);
  shape.closePath();
  const geometry = new THREE.ExtrudeGeometry(shape, { depth: 3, bevelEnabled: false });
  geometry.rotateX(Math.PI / 2);
  const land = new THREE.Mesh(geometry, mat(colors.cream));
  land.position.y = 3;
  land.receiveShadow = true;
  world.add(land);

  const northShape = new THREE.Shape();
  northShape.moveTo(-180, -205);
  northShape.lineTo(-48, -220);
  northShape.lineTo(-45, -160);
  northShape.lineTo(-76, -135);
  northShape.lineTo(-150, -132);
  northShape.closePath();
  const northGeometry = new THREE.ExtrudeGeometry(northShape, { depth: 4, bevelEnabled: false });
  northGeometry.rotateX(Math.PI / 2);
  const north = new THREE.Mesh(northGeometry, mat(0x8f9b67));
  north.position.y = 4;
  north.receiveShadow = true;
  world.add(north);

}

function createRoads() {
  const roads = [
    ...roadX.map(x => [x, 7.05, 112, 9, 240]),
    ...roadZ.map(z => [20, 7.07, z, 400, 9])
  ];
  roads.forEach(([x, y, z, sx, sz]) => {
    box(colors.road, [x, y, z], [sx, 0.18, sz], world);
    if (sx > sz) {
      for (let k = -170; k < 210; k += 20) box(colors.yellow, [k, y + 0.12, z], [7, 0.04, 0.32], world);
    } else {
      for (let k = 0; k < 235; k += 20) box(colors.yellow, [x, y + 0.12, k], [0.32, 0.04, 7], world);
    }
  });
  roadX.forEach(x => {
    roadZ.forEach(z => {
      box(colors.paper, [x, 7.22, z - 6.2], [7.4, 0.04, 2.1], world);
      box(colors.paper, [x, 7.22, z + 6.2], [7.4, 0.04, 2.1], world);
    });
  });
}

function createBuilding(x, z, width, depth, height, color, roof = "flat") {
  const group = new THREE.Group();
  group.position.set(x, 7.65, z);
  box(color, [0, height / 2, 0], [width, height, depth], group, true);
  if (roof === "peak") {
    const top = cone(colors.red, [0, height + 2, 0], [width * 0.66, 4, depth * 0.66], group, true);
    top.rotation.y = Math.PI / 4;
  } else {
    box(colors.ink, [0, height + 0.3, 0], [width * 0.86, 0.6, depth * 0.86], group);
  }
  const windowColor = seeded(x + z) > 0.5 ? colors.yellow : colors.blue;
  const rows = Math.min(4, Math.max(1, Math.floor(height / 8)));
  for (let row = 0; row < rows; row += 1) {
    box(windowColor, [0, 4.6 + row * 6.4, -depth / 2 - 0.06], [width * 0.7, 1.55, 0.16], group);
  }
  box(0x4d4033, [0, 1.9, -depth / 2 - 0.12], [2.1, 3.8, 0.24], group);
  if (seeded(x * 0.4 + z) > 0.48) {
    const awning = box(colors.orange, [0, 6.5, -depth / 2 - 0.65], [width * 0.65, 0.45, 1.5], group, true);
    awning.rotation.x = -0.18;
  }
  world.add(group);
  buildingBounds.push({
    minX: x - width / 2 - 1.2,
    maxX: x + width / 2 + 1.2,
    minZ: z - depth / 2 - 1.2,
    maxZ: z + depth / 2 + 1.2,
    height: 7.65 + height
  });
  return group;
}

function createCity() {
  const palette = [0xd9b59d, 0xc6b18c, 0xe2c9a3, 0xb9b58e, 0xcc8c79, 0xaec0a4];
  const xStops = [-178, ...roadX, 218];
  const zStops = [-8, ...roadZ, 236];
  let index = 1;
  for (let zi = 0; zi < zStops.length - 1; zi += 1) {
    for (let xi = 0; xi < xStops.length - 1; xi += 1) {
      const lowX = xStops[xi] + (xi === 0 ? 1 : 6);
      const highX = xStops[xi + 1] - (xi === xStops.length - 2 ? 1 : 6);
      const lowZ = zStops[zi] + (zi === 0 ? 1 : 6);
      const highZ = zStops[zi + 1] - (zi === zStops.length - 2 ? 1 : 6);
      const centerX = (lowX + highX) / 2;
      const centerZ = (lowZ + highZ) / 2;
      const blockWidth = highX - lowX;
      const blockDepth = highZ - lowZ;
      const landmarkSpace =
        centerX > 35 && centerX < 100 && centerZ > 40 && centerZ < 112 ||
        centerX > 100 && centerZ < 112 ||
        centerX < 5 && centerZ > 125 ||
        centerX < -105 && centerZ < 38;
      box(index % 2 ? 0xd6c99f : 0xc9c294, [centerX, 7.23, centerZ], [blockWidth, 0.36, blockDepth], world, true);
      if (landmarkSpace || seeded(index) < 0.14) {
        index += 1;
        continue;
      }
      const buildingCount = blockWidth > 46 ? 2 : 1;
      for (let buildingIndex = 0; buildingIndex < buildingCount; buildingIndex += 1) {
        const sectionWidth = blockWidth / buildingCount;
        const width = Math.min(22, sectionWidth - 5);
        const depth = Math.min(27, blockDepth - 7);
        const x = lowX + sectionWidth * (buildingIndex + 0.5);
        const z = centerZ + (seeded(index + buildingIndex) - 0.5) * 3;
        const financial = centerX > 80 && centerZ < 145;
        const height = financial ? 24 + seeded(index + 3) * 24 : 12 + seeded(index + 3) * 14;
        createBuilding(x, z, width, depth, height, palette[(index + buildingIndex) % palette.length], seeded(index + 4) > 0.82 ? "peak" : "flat");
      }
      index += 1;
    }
  }
}

function cableBetween(start, end, sag, color = colors.ink, thickness = 0.2) {
  const midpoint = start.clone().add(end).multiplyScalar(0.5);
  midpoint.y -= sag;
  const curve = new THREE.QuadraticBezierCurve3(start, midpoint, end);
  const geometry = new THREE.TubeGeometry(curve, 18, thickness, 5, false);
  const cable = new THREE.Mesh(geometry, mat(color));
  world.add(cable);
}

function createBridge() {
  box(colors.bridge, [-121, 21, -94], [17, 2.2, 166], world, true);
  box(colors.road, [-121, 22.2, -94], [11.5, 0.25, 166], world);
  const towerZ = [-46, -142];
  towerZ.forEach(z => {
    [-127, -115].forEach(x => box(colors.bridge, [x, 43, z], [3.6, 44, 5], world, true));
    box(colors.bridge, [-121, 62, z], [17, 3.6, 5], world, true);
    box(colors.bridge, [-121, 47, z], [17, 2.2, 5], world, true);
    box(colors.bridge, [-121, 33, z], [17, 2.2, 5], world, true);
  });
  [-127, -115].forEach(x => {
    cableBetween(new THREE.Vector3(x, 62, -46), new THREE.Vector3(x, 62, -142), 28, colors.bridge, 0.32);
    cableBetween(new THREE.Vector3(x, 62, -46), new THREE.Vector3(x, 24, -12), 4, colors.bridge, 0.26);
    cableBetween(new THREE.Vector3(x, 62, -142), new THREE.Vector3(x, 25, -186), 6, colors.bridge, 0.26);
    for (let z = -52; z >= -136; z -= 9) {
      const ratio = (z + 46) / -96;
      const cableY = 62 - Math.sin(ratio * Math.PI) * 28;
      cableBetween(new THREE.Vector3(x, cableY, z), new THREE.Vector3(x, 23, z), 0, colors.bridge, 0.1);
    }
  });
  box(0x9a956d, [-138, 9, -18], [30, 14, 28], world);
  cone(colors.darkGreen, [-156, 21, -12], [40, 28, 38], world);
}

function createPier() {
  box(colors.wood, [22, 3.4, -48], [42, 2.2, 63], world, true);
  for (let x = 4; x <= 40; x += 12) {
    for (let z = -77; z <= -22; z += 18) cylinder(colors.wood, [x, 1.1, z], [1.2, 5, 1.2], world);
  }
  createBuilding(22, -38, 27, 24, 13, 0xe0c79f, "peak");
  cylinder(colors.orange, [22, 18, -38], [4.3, 4, 4.3], world, true);
  const wheel = new THREE.Group();
  wheel.position.set(22, 29, -39);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(10, 0.65, 7, 22), mat(colors.red));
  wheel.add(ring);
  for (let i = 0; i < 8; i += 1) {
    const angle = i * Math.PI / 4;
    const spoke = box(colors.ink, [Math.cos(angle) * 5, Math.sin(angle) * 5, 0], [0.28, 9.5, 0.28], wheel);
    spoke.rotation.z = angle - Math.PI / 2;
  }
  world.add(wheel);

  box(colors.wood, [95, 3.2, -38], [10, 1.7, 75], world, true);
  for (let z = -68; z < -8; z += 14) {
    cylinder(colors.wood, [92, 0.6, z], [0.7, 6, 0.7], world);
    cylinder(colors.wood, [98, 0.6, z], [0.7, 6, 0.7], world);
  }
}

function createChinatownGate() {
  const gate = new THREE.Group();
  gate.position.set(59, 8, 76);
  box(colors.red, [-7, 7, 0], [2.2, 15, 2.2], gate, true);
  box(colors.red, [7, 7, 0], [2.2, 15, 2.2], gate, true);
  box(colors.red, [0, 14, 0], [17, 2, 2.5], gate, true);
  const roof = cone(colors.darkGreen, [0, 17, 0], [14, 4, 6], gate, true);
  roof.rotation.y = Math.PI / 4;
  [-8.5, 8.5].forEach(x => sphere(colors.yellow, [x, 15, 0], [1.4, 1.4, 1.4], gate));
  for (let x = -14; x <= 14; x += 7) {
    sphere(colors.red, [x, 12, -6], [1.4, 1.4, 1.4], gate);
    cylinder(colors.yellow, [x, 10.5, -6], [0.12, 1.4, 0.12], gate);
  }
  world.add(gate);
}

function createCoitTower() {
  const tower = new THREE.Group();
  tower.position.set(126, 8, 54);
  cylinder(colors.cream, [0, 16, 0], [5.5, 32, 5.5], tower, true);
  cylinder(colors.paper, [0, 33, 0], [6.8, 3, 6.8], tower, true);
  for (let i = 0; i < 6; i += 1) {
    const angle = i * Math.PI / 3;
    box(colors.ink, [Math.cos(angle) * 5.2, 26, Math.sin(angle) * 5.2], [1.1, 6, 0.35], tower);
  }
  world.add(tower);
}

function createFerryBuilding() {
  const group = new THREE.Group();
  group.position.set(148, 7.65, 4);
  box(0xc99f76, [0, 6, 0], [52, 12, 15], group, true);
  box(colors.cream, [0, 17, 0], [10, 22, 10], group, true);
  box(colors.ink, [0, 29, 0], [12, 2, 12], group, true);
  const roof = cone(colors.cream, [0, 35, 0], [11, 12, 11], group, true);
  roof.rotation.y = Math.PI / 4;
  const clockFace = new THREE.Mesh(new THREE.CircleGeometry(3.1, 18), mat(colors.paper));
  clockFace.position.set(0, 21, -5.08);
  group.add(clockFace);
  const clockRing = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(
    Array.from({ length: 19 }, (_, index) => {
      const angle = index / 18 * Math.PI * 2;
      return new THREE.Vector3(Math.cos(angle) * 3.1, Math.sin(angle) * 3.1 + 21, -5.12);
    })
  ), inkMaterial);
  group.add(clockRing);
  box(colors.ink, [0, 22, -5.18], [0.22, 2.1, 0.12], group);
  box(colors.ink, [0.9, 21, -5.18], [1.8, 0.22, 0.12], group);
  for (let x = -21; x <= 21; x += 7) {
    box(colors.blue, [x, 6, -7.56], [2.2, 4, 0.18], group);
  }
  world.add(group);
}

function createTransamerica() {
  const group = new THREE.Group();
  group.position.set(146, 7.65, 88);
  const geometry = new THREE.ConeGeometry(15, 72, 4);
  const tower = new THREE.Mesh(geometry, mat(0xd9d0b3));
  tower.position.y = 36;
  tower.rotation.y = Math.PI / 4;
  tower.castShadow = true;
  group.add(tower);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 18), inkMaterial);
  edges.scale.setScalar(1.003);
  tower.add(edges);
  box(0xd9d0b3, [-9, 35, 0], [13, 5, 5], group, true);
  box(0xd9d0b3, [9, 35, 0], [13, 5, 5], group, true);
  world.add(group);
}

function createPaintedLadies() {
  const houseColors = [0xd78779, 0x7fa696, 0xd3ad61, 0x8d91ad];
  [-128, -114, -100, -86].forEach((x, index) => {
    const house = new THREE.Group();
    house.position.set(x, 7.65, 143 + index * 0.7);
    box(houseColors[index], [0, 8, 0], [11, 16, 18], house, true);
    const roof = cone(colors.ink, [0, 19, 0], [10, 7, 13], house, true);
    roof.rotation.y = Math.PI / 4;
    box(colors.paper, [0, 6, -9.12], [7.5, 6.5, 0.3], house, true);
    box(colors.ink, [0, 3.1, -9.35], [2.2, 5.8, 0.22], house);
    [-3, 3].forEach(windowX => {
      box(colors.blue, [windowX, 11, -9.25], [2.2, 3.1, 0.2], house);
    });
    cylinder(colors.paper, [-4.2, 3, -9.6], [0.4, 6, 0.4], house);
    cylinder(colors.paper, [4.2, 3, -9.6], [0.4, 6, 0.4], house);
    world.add(house);
  });
}

function createAlcatraz() {
  const island = sphere(0x8a8d63, [78, 1.1, -150], [72, 8, 48], world, true);
  island.rotation.y = -0.2;
  box(0xb9ad8b, [78, 8, -150], [42, 12, 19], world, true);
  box(colors.ink, [78, 14.3, -150], [38, 0.7, 16], world);
  const lighthouse = new THREE.Group();
  lighthouse.position.set(52, 3.5, -145);
  cylinder(colors.cream, [0, 9, 0], [4, 18, 4], lighthouse, true);
  cylinder(colors.red, [0, 18.7, 0], [5.2, 2.2, 5.2], lighthouse, true);
  cone(colors.red, [0, 22, 0], [5, 4.5, 5], lighthouse, true);
  world.add(lighthouse);
}

function createWharfDetails() {
  const sign = makeTextSprite("PIER 39", "#ef5b3f");
  sign.position.set(22, 38, -57);
  sign.scale.set(25, 6.25, 1);
  world.add(sign);
  const arch = new THREE.Group();
  arch.position.set(22, 5, -15);
  cylinder(colors.wood, [-10, 7, 0], [1.2, 14, 1.2], arch, true);
  cylinder(colors.wood, [10, 7, 0], [1.2, 14, 1.2], arch, true);
  box(colors.red, [0, 14, 0], [23, 3.5, 2], arch, true);
  world.add(arch);
}

function createTree(x, z, size = 1) {
  const group = new THREE.Group();
  group.position.set(x, 7.5, z);
  cylinder(colors.wood, [0, 3 * size, 0], [1.1 * size, 6 * size, 1.1 * size], group);
  sphere(seeded(x + z) > 0.45 ? colors.green : colors.darkGreen, [0, 8 * size, 0], [6 * size, 8 * size, 6 * size], group, true);
  world.add(group);
}

function createVegetation() {
  for (let i = 0; i < 52; i += 1) {
    const x = -170 + seeded(i * 3) * 360;
    const z = 12 + seeded(i * 3 + 1) * 215;
    const roadClear = roadX.every(value => Math.abs(x - value) > 8) && roadZ.every(value => Math.abs(z - value) > 8);
    const buildingClear = buildingBounds.every(bounds =>
      x < bounds.minX - 5 || x > bounds.maxX + 5 || z < bounds.minZ - 5 || z > bounds.maxZ + 5
    );
    const landmarkClear =
      !(x > 118 && z < 112) &&
      !(x > 32 && x < 96 && z > 45 && z < 112) &&
      !(x < -75 && x > -138 && z > 128 && z < 158);
    if (roadClear && buildingClear && landmarkClear && seeded(i + 9) > 0.42) createTree(x, z, 0.65 + seeded(i + 7) * 0.7);
  }
  for (let i = 0; i < 18; i += 1) {
    createTree(-170 + seeded(i + 50) * 110, -205 + seeded(i + 70) * 75, 0.9 + seeded(i + 90) * 0.8);
  }
}

function createCloud(x, y, z, scale) {
  const cloud = new THREE.Group();
  cloud.position.set(x, y, z);
  [[0, 0, 0, 8], [8, 1, 1, 6], [-8, 0, 2, 6], [2, 3, 0, 7]].forEach(part => {
    sphere(colors.paper, [part[0], part[1], part[2]], [part[3], part[3] * 0.52, part[3] * 0.7], cloud);
  });
  cloud.scale.setScalar(scale);
  cloud.userData.speed = 0.8 + scale * 0.3;
  world.add(cloud);
  clouds.push(cloud);
}

function makeTextSprite(text, color = "#1c2b2a") {
  const textCanvas = document.createElement("canvas");
  textCanvas.width = 512;
  textCanvas.height = 128;
  const context = textCanvas.getContext("2d");
  context.fillStyle = "#f4e8ca";
  context.fillRect(0, 19, 512, 88);
  context.strokeStyle = "#1c2b2a";
  context.lineWidth = 8;
  context.strokeRect(4, 23, 504, 80);
  context.fillStyle = color;
  context.font = "bold 42px Georgia";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, 256, 64);
  const texture = new THREE.CanvasTexture(textCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
  sprite.scale.set(26, 6.5, 1);
  return sprite;
}

function createLabels() {
  [
    ["GOLDEN GATE", -121, 73, -95],
    ["FISHERMAN'S WHARF", 22, 46, -32],
    ["CHINATOWN", 59, 42, 78],
    ["FERRY BUILDING", 148, 51, 4],
    ["TRANSAMERICA", 146, 86, 88],
    ["PAINTED LADIES", -107, 37, 143],
    ["ALCATRAZ", 78, 28, -150],
    ["GOLDEN GATE PARK", -40, 29, 194]
  ].forEach(([text, x, y, z]) => {
    const label = makeTextSprite(text);
    label.position.set(x, y, z);
    world.add(label);
  });
}

const entities = [];

function registerEntity(group, type, radius, update) {
  entities.push({ group, type, radius, update, hitAt: -10 });
  return group;
}

function createPerson(x, z, hue, route) {
  const group = new THREE.Group();
  group.position.set(x, 7.5, z);
  const skin = [0x8f604b, 0xd3a281, 0x6f4837, 0xe0b48f][Math.floor(seeded(x * z) * 4)];
  const pants = [0x263f54, 0x483b48, 0x36574f][Math.floor(seeded(x + z) * 3)];
  const torso = box(hue, [0, 4.1, 0], [2.8, 3.7, 1.7], group, true);
  torso.rotation.z = (seeded(z) - 0.5) * 0.04;
  sphere(skin, [0, 7.1, -0.05], [2.5, 2.7, 2.4], group, true);
  const hair = sphere(colors.ink, [0, 7.9, 0.18], [2.55, 1.5, 2.4], group);
  const nose = sphere(skin, [0, 6.9, -1.25], [0.55, 0.7, 0.65], group);
  hair.scale.z = 0.92;
  nose.scale.z = 1.2;
  const leftLeg = new THREE.Group();
  const rightLeg = new THREE.Group();
  leftLeg.position.set(-0.7, 2.35, 0);
  rightLeg.position.set(0.7, 2.35, 0);
  cylinder(pants, [0, -1.15, 0], [0.62, 2.3, 0.62], leftLeg);
  cylinder(pants, [0, -1.15, 0], [0.62, 2.3, 0.62], rightLeg);
  box(colors.ink, [0, -2.35, -0.25], [0.85, 0.45, 1.4], leftLeg);
  box(colors.ink, [0, -2.35, -0.25], [0.85, 0.45, 1.4], rightLeg);
  const leftArm = new THREE.Group();
  const rightArm = new THREE.Group();
  leftArm.position.set(-1.65, 5.25, 0);
  rightArm.position.set(1.65, 5.25, 0);
  cylinder(skin, [0, -1.25, 0], [0.48, 2.5, 0.48], leftArm);
  cylinder(skin, [0, -1.25, 0], [0.48, 2.5, 0.48], rightArm);
  group.add(leftLeg, rightLeg, leftArm, rightArm);
  if (seeded(x - z) > 0.56) {
    cylinder(colors.yellow, [0, 8.55, 0], [1.4, 0.45, 1.4], group);
    box(colors.yellow, [0, 8.42, -1.1], [3.5, 0.22, 1.1], group);
  }
  if (seeded(x + z * 2) > 0.6) box(colors.red, [1.65, 3.25, 0.2], [1, 2.6, 1.8], group);
  world.add(group);
  const axis = route.axis;
  const origin = group.position[axis];
  const phase = route.phase;
  return registerEntity(group, "person", 3.3, time => {
    const stride = time * 0.62 + phase;
    group.position[axis] = origin + Math.sin(stride) * route.distance;
    const direction = Math.cos(stride);
    group.rotation.y = axis === "z" ? direction > 0 ? Math.PI : 0 : direction > 0 ? -Math.PI / 2 : Math.PI / 2;
    leftLeg.rotation.x = Math.sin(stride * 2) * 0.42;
    rightLeg.rotation.x = -leftLeg.rotation.x;
    leftArm.rotation.x = -leftLeg.rotation.x * 0.8;
    rightArm.rotation.x = leftLeg.rotation.x * 0.8;
    group.position.y = 7.5 + Math.abs(Math.sin(stride * 2)) * 0.08;
  });
}

function createCar(x, z, color, axis = "x", speed = 8) {
  const group = new THREE.Group();
  group.position.set(x, 8.5, z);
  box(color, [0, 1.3, 0], [7, 2.2, 3.8], group, true);
  box(colors.paper, [0.8, 3, 0], [3.5, 1.7, 3.3], group, true);
  [-2.2, 2.2].forEach(wx => {
    [-2, 2].forEach(wz => {
      const wheel = new THREE.Mesh(geometryCache.wheel, mat(colors.ink));
      wheel.position.set(wx, 0.4, wz);
      wheel.rotation.x = Math.PI / 2;
      group.add(wheel);
    });
  });
  if (axis === "z") group.rotation.y = Math.PI / 2;
  world.add(group);
  const start = group.position[axis];
  const minimum = axis === "x" ? -168 : -2;
  const maximum = axis === "x" ? 208 : 232;
  const range = maximum - minimum;
  return registerEntity(group, "car", 4.6, time => {
    group.position[axis] = ((time * speed + start - minimum) % range + range) % range + minimum;
  });
}

function createCableCar() {
  const group = new THREE.Group();
  group.position.set(-44, 9.5, 110);
  box(colors.yellow, [0, 2.2, 0], [5.5, 4.5, 10], group, true);
  box(colors.red, [0, 4.8, 0], [5.8, 0.8, 10.4], group, true);
  for (let z = -3.2; z <= 3.2; z += 3.2) {
    box(colors.blue, [-2.8, 2.8, z], [0.18, 1.8, 2], group);
    box(colors.blue, [2.8, 2.8, z], [0.18, 1.8, 2], group);
  }
  cylinder(colors.ink, [0, 9.4, 0], [0.12, 9.2, 0.12], group);
  cableBetween(new THREE.Vector3(-44, 19, -10), new THREE.Vector3(-44, 19, 215), 8, colors.ink, 0.08);
  world.add(group);
  return registerEntity(group, "car", 5.5, time => {
    group.position.z = 105 + Math.sin(time * 0.13) * 100;
    group.rotation.y = Math.cos(time * 0.13) > 0 ? 0 : Math.PI;
  });
}

function createAnimal(x, z, kind = "dog") {
  const group = new THREE.Group();
  group.position.set(x, kind === "seal" ? 1.4 : 7.6, z);
  const color = kind === "seal" ? 0x6e6455 : kind === "cat" ? 0xb77443 : 0x8b6c4e;
  sphere(color, [0, 1.1, 0], [3.8, 2.2, 2.1], group, true);
  sphere(color, [2.4, 2.1, 0], [2, 2, 1.8], group, true);
  if (kind === "seal") {
    cone(color, [-3.4, 0.9, 0], [2.2, 3.6, 1.2], group).rotation.z = Math.PI / 2;
  } else {
    [-1.4, 1.3].forEach(px => {
      [-1, 1].forEach(pz => cylinder(color, [px, -0.4, pz], [0.35, 1.7, 0.35], group));
    });
    const tail = cylinder(color, [-3, 2.2, 0], [0.35, 4.5, 0.35], group);
    tail.rotation.z = -0.75;
  }
  world.add(group);
  const originX = x;
  const phase = seeded(x - z) * Math.PI * 2;
  return registerEntity(group, "animal", 3.6, time => {
    if (kind !== "seal") group.position.x = originX + Math.sin(time * 0.45 + phase) * 8;
    group.rotation.y = Math.cos(time * 0.45 + phase) > 0 ? 0 : Math.PI;
  });
}

function populateEntities() {
  const peopleColors = [colors.orange, colors.blue, colors.green, colors.pink, colors.yellow];
  [
    [-137, 42, "z", 15], [-72, 42, "z", 15], [-7, 42, "z", 15],
    [58, 42, "z", 15], [123, 42, "z", 15], [172, 42, "z", 15],
    [-112, 60, "x", 23], [-47, 76, "x", 23], [18, 110, "x", 23],
    [83, 126, "x", 23], [148, 160, "x", 22], [-137, 143, "z", 16],
    [-72, 194, "z", 14], [8, -35, "x", 11], [36, -59, "x", 10],
    [91, -30, "z", 14]
  ].forEach(([x, z, axis, distance], index) => createPerson(
    x,
    z,
    peopleColors[index % peopleColors.length],
    { axis, distance, phase: index * 0.83 }
  ));
  [
    [-80, 18, colors.red, "x", 7],
    [50, 68, colors.blue, "x", 9],
    [120, 118, colors.orange, "x", 10],
    [-15, 160, colors.green, "z", 7],
    [50, 94, colors.yellow, "z", 8],
    [-145, 120, colors.paper, "z", 11]
  ].forEach(item => createCar(...item));
  createCableCar();
  createAnimal(-74, 164, "dog");
  createAnimal(67, 116, "cat");
  createAnimal(12, -71, "seal");
  createAnimal(26, -72, "seal");
  createAnimal(42, -69, "seal");
}

const clouds = [];
createLand();
createRoads();
createCity();
createBridge();
createPier();
createChinatownGate();
createCoitTower();
createFerryBuilding();
createTransamerica();
createPaintedLadies();
createAlcatraz();
createWharfDetails();
createVegetation();
createLabels();
populateEntities();
createCloud(-100, 95, -80, 1.2);
createCloud(80, 115, 40, 0.9);
createCloud(160, 86, -150, 1.5);

function createBird() {
  const bird = new THREE.Group();
  sphere(colors.paper, [0, 0, 0.8], [5.2, 3.5, 8.4], bird, true);
  sphere(0xd3d4c8, [0, 1.75, 1.8], [4.4, 1.25, 5.2], bird);
  sphere(colors.paper, [0, 0.7, -3.4], [4.1, 3.1, 3.8], bird);
  sphere(colors.paper, [0, 1.4, -5.3], [3.5, 3.2, 3.7], bird, true);
  const beak = cone(colors.orange, [0, 0.75, -8.5], [1.55, 4.3, 1.55], bird, true);
  beak.rotation.x = -Math.PI / 2;
  [-1.42, 1.42].forEach(x => {
    sphere(colors.ink, [x, 2.15, -6.25], [0.28, 0.34, 0.24], bird);
  });
  const createWing = direction => {
    const wing = new THREE.Group();
    wing.position.set(direction * 2.55, 0.9, -0.4);
    const points = [
      new THREE.Vector3(0, 0, -2.4),
      new THREE.Vector3(direction * 13.5, 0, -0.3),
      new THREE.Vector3(direction * 11.2, 0, 2.4),
      new THREE.Vector3(direction * 8, 0, 5.5),
      new THREE.Vector3(direction * 1.4, 0, 4.5)
    ];
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute([
      ...points[0].toArray(), ...points[1].toArray(), ...points[2].toArray(),
      ...points[0].toArray(), ...points[2].toArray(), ...points[3].toArray(),
      ...points[0].toArray(), ...points[3].toArray(), ...points[4].toArray()
    ], 3));
    geometry.computeVertexNormals();
    wing.add(new THREE.Mesh(geometry, mat(colors.paper, { side: THREE.DoubleSide })));
    const outline = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), inkMaterial);
    outline.position.y = 0.05;
    wing.add(outline);
    const tipPoints = [
      points[1].clone().setY(0.08),
      points[2].clone().setY(0.08),
      new THREE.Vector3(direction * 9.3, 0.08, 3.8),
      new THREE.Vector3(direction * 10.5, 0.08, 0.2)
    ];
    const tipGeometry = new THREE.BufferGeometry();
    tipGeometry.setAttribute("position", new THREE.Float32BufferAttribute([
      ...tipPoints[0].toArray(), ...tipPoints[1].toArray(), ...tipPoints[2].toArray(),
      ...tipPoints[0].toArray(), ...tipPoints[2].toArray(), ...tipPoints[3].toArray()
    ], 3));
    tipGeometry.computeVertexNormals();
    wing.add(new THREE.Mesh(tipGeometry, mat(colors.ink, { side: THREE.DoubleSide })));
    return wing;
  };
  const leftWing = createWing(-1);
  const rightWing = createWing(1);
  bird.add(leftWing, rightWing);
  [-1.8, 0, 1.8].forEach((x, index) => {
    const feather = cone(colors.paper, [x, -0.2, 7.4 + (index === 1 ? 1 : 0)], [1.5, 5.4, 1.15], bird, true);
    feather.rotation.x = Math.PI / 2;
    feather.rotation.z = (index - 1) * 0.12;
  });
  const legs = new THREE.Group();
  [-1.15, 1.15].forEach(x => {
    cylinder(colors.orange, [x, -3.2, 2.3], [0.34, 2.4, 0.34], legs);
    const foot = new THREE.Group();
    foot.position.set(x, -4.4, 1.7);
    [-0.45, 0, 0.45].forEach(toeX => {
      const toe = cylinder(colors.orange, [toeX, 0, -0.7], [0.13, 1.5, 0.13], foot);
      toe.rotation.x = Math.PI / 2;
    });
    legs.add(foot);
  });
  bird.add(legs);
  bird.scale.setScalar(0.51);
  bird.userData.leftWing = leftWing;
  bird.userData.rightWing = rightWing;
  bird.userData.legs = legs;
  scene.add(bird);
  return bird;
}

const bird = createBird();
bird.position.set(-88, 31, 12);

const state = {
  started: false,
  paused: false,
  yaw: Math.PI,
  speed: 18,
  verticalSpeed: 0,
  mode: "Flying",
  score: 0,
  streak: 0,
  lastHit: -10,
  lastPoop: -10,
  lastTakeoff: -10,
  sound: true,
  mission: { person: 0, car: 0, animal: 0 },
  complete: false
};

const keys = {};
const mobile = { x: 0, y: 0, flap: false, dive: false };
const poops = [];
const splats = [];
let toastTimer;
let audioContext;

function audioTone(frequency, duration, type = "triangle", volume = 0.04) {
  if (!state.sound) return;
  audioContext ||= new AudioContext();
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  oscillator.type = type;
  oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
  oscillator.frequency.exponentialRampToValueAtTime(frequency * 0.68, audioContext.currentTime + duration);
  gain.gain.setValueAtTime(volume, audioContext.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.0001, audioContext.currentTime + duration);
  oscillator.connect(gain).connect(audioContext.destination);
  oscillator.start();
  oscillator.stop(audioContext.currentTime + duration);
}

function showToast(message) {
  const toast = document.querySelector("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 1150);
}

function groundAt(x, z) {
  if (Math.abs(x + 121) < 9 && z > -178 && z < -10) return { height: 22.5, surface: "land" };
  if (x > 0 && x < 44 && z > -80 && z < -16) return { height: 4.7, surface: "land" };
  if (x > 89 && x < 101 && z > -78 && z < 0) return { height: 4.3, surface: "land" };
  const city = x > -182 && x < 220 && z > -22 && z < 240;
  const north = x > -180 && x < -45 && z > -215 && z < -130;
  return city || north ? { height: 7.4, surface: "land" } : { height: 0.8, surface: "water" };
}

function buildingAt(x, z) {
  return buildingBounds.find(bounds =>
    x > bounds.minX && x < bounds.maxX && z > bounds.minZ && z < bounds.maxZ
  );
}

function dropPoop() {
  const time = clock.elapsedTime;
  if (!state.started || state.paused || time - state.lastPoop < 0.42) return;
  state.lastPoop = time;
  const poop = sphere(0xf7f2dd, [0, 0, 0], [0.55, 0.75, 0.55], scene, true);
  poop.position.copy(bird.position);
  poop.position.y -= 1.5;
  const forward = new THREE.Vector3(Math.sin(state.yaw), 0, Math.cos(state.yaw));
  poop.userData.velocity = forward.multiplyScalar(state.speed * 0.35);
  poop.userData.velocity.y = -7;
  poop.userData.born = time;
  poops.push(poop);
  audioTone(420, 0.08, "square", 0.018);
}

function makeSplat(position, surface) {
  const geometry = new THREE.CircleGeometry(1.6, 12);
  const splat = new THREE.Mesh(geometry, new THREE.MeshToonMaterial({
    color: 0xf5f0d9,
    transparent: true
  }));
  splat.position.copy(position);
  if (surface === "water") {
    splat.rotation.x = -Math.PI / 2;
    splat.position.y = 0.9;
  } else {
    splat.rotation.x = -Math.PI / 2;
    splat.position.y += 0.08;
  }
  splat.scale.set(1.2, 0.65, 1);
  scene.add(splat);
  splats.push({ mesh: splat, born: clock.elapsedTime });
}

function registerHit(entity, position) {
  const time = clock.elapsedTime;
  if (time - entity.hitAt < 3) return;
  entity.hitAt = time;
  state.streak = time - state.lastHit < 5 ? state.streak + 1 : 1;
  state.lastHit = time;
  const points = 10 * state.streak;
  state.score += points;
  if (state.mission[entity.type] < ({ person: 3, car: 2, animal: 1 })[entity.type]) state.mission[entity.type] += 1;
  const labels = {
    person: ["Direct hit!", "Hat hazard!", "Tourist tagged!"],
    car: ["Fresh paint!", "Windshield special!", "Parking violation!"],
    animal: ["Nature answers back!", "Wildlife marked!", "Food chain!"]
  };
  const label = labels[entity.type][Math.floor(Math.random() * labels[entity.type].length)];
  showToast(`${label} +${points}`);
  audioTone(690 + state.streak * 55, 0.24, "triangle", 0.06);
  makeSplat(position, "land");
  entity.group.rotation.z = 0.18;
  updateHud();
  if (missionTotal() === 6 && !state.complete) {
    state.complete = true;
    setTimeout(() => {
      state.paused = true;
      document.querySelector("#final-score").textContent = `You caused ${state.score} civic complaints.`;
      document.querySelector("#complete-screen").classList.remove("hidden");
    }, 900);
  }
}

function missionTotal() {
  return state.mission.person + state.mission.car + state.mission.animal;
}

function updatePoops(delta) {
  for (let i = poops.length - 1; i >= 0; i -= 1) {
    const poop = poops[i];
    poop.userData.velocity.y -= 24 * delta;
    poop.position.addScaledVector(poop.userData.velocity, delta);
    poop.rotation.x += delta * 8;
    poop.rotation.z += delta * 5;
    let hit = false;
    for (const entity of entities) {
      const target = entity.group.position;
      const dx = poop.position.x - target.x;
      const dz = poop.position.z - target.z;
      const dy = poop.position.y - (target.y + 3);
      if (dx * dx + dz * dz < entity.radius * entity.radius && Math.abs(dy) < 4.5) {
        registerHit(entity, poop.position.clone());
        hit = true;
        break;
      }
    }
    const ground = groundAt(poop.position.x, poop.position.z);
    if (!hit && poop.position.y <= ground.height) {
      makeSplat(new THREE.Vector3(poop.position.x, ground.height, poop.position.z), ground.surface);
      hit = true;
    }
    if (hit || clock.elapsedTime - poop.userData.born > 6) {
      scene.remove(poop);
      poops.splice(i, 1);
    }
  }
  for (let i = splats.length - 1; i >= 0; i -= 1) {
    const splat = splats[i];
    if (clock.elapsedTime - splat.born > 12) {
      splat.mesh.material.transparent = true;
      splat.mesh.material.opacity -= delta * 0.7;
      if (splat.mesh.material.opacity <= 0) {
        scene.remove(splat.mesh);
        splats.splice(i, 1);
      }
    }
  }
}

function updateBird(delta, elapsed) {
  const ground = groundAt(bird.position.x, bird.position.z);
  const previousPosition = bird.position.clone();
  const forwardInput = (keys.KeyW ? 1 : 0) - (keys.KeyS ? 1 : 0) - mobile.y;
  const turnInput = (keys.KeyD ? 1 : 0) - (keys.KeyA ? 1 : 0) + mobile.x;
  const wantsFlap = keys.Space || mobile.flap;
  let flying = bird.position.y > ground.height + 2.3 || state.verticalSpeed > 1;
  const takingOff = !flying && wantsFlap;
  if (takingOff) {
    flying = true;
    state.verticalSpeed = 10.8;
    bird.position.y += 0.45;
    if (elapsed - state.lastTakeoff > 0.8) {
      state.lastTakeoff = elapsed;
      audioTone(310, 0.14, "triangle", 0.035);
    }
  }
  if (flying) {
    state.mode = "Flying";
    state.speed += (14 + Math.max(0, forwardInput) * 13 - state.speed) * delta * 1.8;
    state.yaw -= turnInput * delta * 1.45;
    state.verticalSpeed -= 6.2 * delta;
    if (wantsFlap && !takingOff) {
      state.verticalSpeed += 18 * delta;
      state.verticalSpeed = Math.min(state.verticalSpeed, 8.5);
    }
    if (keys.ShiftLeft || keys.ShiftRight || mobile.dive) state.verticalSpeed -= 13 * delta;
    state.verticalSpeed = Math.max(-11, state.verticalSpeed);
    bird.position.y += state.verticalSpeed * delta;
  } else if (ground.surface === "water") {
    state.mode = "Swimming";
    state.speed += (4.2 + Math.max(0, forwardInput) * 3.5 - state.speed) * delta * 3;
    state.yaw -= turnInput * delta * 2;
    bird.position.y = ground.height + 0.9 + Math.sin(elapsed * 2.2) * 0.1;
    state.verticalSpeed = 0;
  } else {
    state.mode = "Walking";
    state.speed += ((forwardInput > 0 ? 6.2 : 1.2) - state.speed) * delta * 4;
    state.yaw -= turnInput * delta * 2.3;
    bird.position.y = ground.height + 1.7;
    state.verticalSpeed = 0;
  }

  const forward = new THREE.Vector3(Math.sin(state.yaw), 0, Math.cos(state.yaw));
  if (flying || forwardInput > 0 || state.mode === "Swimming") bird.position.addScaledVector(forward, state.speed * delta);
  if (!flying && forwardInput < 0) bird.position.addScaledVector(forward, -state.speed * 0.55 * delta);
  const occupiedBuilding = buildingAt(bird.position.x, bird.position.z);
  if (occupiedBuilding && !flying) {
    bird.position.x = previousPosition.x;
    bird.position.z = previousPosition.z;
    state.speed *= 0.25;
  } else if (occupiedBuilding && bird.position.y < occupiedBuilding.height + 2) {
    bird.position.y = occupiedBuilding.height + 2;
    state.verticalSpeed = Math.max(0, state.verticalSpeed);
  }
  bird.position.x = THREE.MathUtils.clamp(bird.position.x, -260, 270);
  bird.position.z = THREE.MathUtils.clamp(bird.position.z, -260, 270);
  const currentGround = groundAt(bird.position.x, bird.position.z);
  bird.position.y = Math.max(currentGround.height + (currentGround.surface === "water" ? 0.9 : 1.6), bird.position.y);
  bird.position.y = Math.min(115, bird.position.y);
  bird.rotation.y = state.yaw + Math.PI;
  bird.rotation.z = THREE.MathUtils.lerp(bird.rotation.z, -turnInput * 0.3, delta * 5);
  bird.rotation.x = THREE.MathUtils.lerp(bird.rotation.x, -state.verticalSpeed * 0.025, delta * 4);
  const flap = state.mode === "Flying" ? Math.sin(elapsed * (wantsFlap ? 15 : 7)) * 0.48 : Math.sin(elapsed * 4) * 0.05;
  bird.userData.leftWing.rotation.z = flap;
  bird.userData.rightWing.rotation.z = -flap;
  const wingSpread = state.mode === "Flying" ? 1 : 0.22;
  bird.userData.leftWing.scale.x = THREE.MathUtils.lerp(bird.userData.leftWing.scale.x, wingSpread, delta * 5);
  bird.userData.rightWing.scale.x = THREE.MathUtils.lerp(bird.userData.rightWing.scale.x, wingSpread, delta * 5);
  bird.userData.legs.rotation.x = THREE.MathUtils.lerp(bird.userData.legs.rotation.x, state.mode === "Flying" ? -1.25 : 0, delta * 6);
  bird.userData.legs.scale.y = THREE.MathUtils.lerp(bird.userData.legs.scale.y, state.mode === "Flying" ? 0.18 : 1, delta * 6);
}

function updateCamera(delta) {
  const ground = groundAt(bird.position.x, bird.position.z);
  const heightFactor = THREE.MathUtils.clamp((bird.position.y - ground.height) / 45, 0, 1);
  const distance = 25 + heightFactor * 10;
  const offset = new THREE.Vector3(-Math.sin(state.yaw) * distance, 12 + heightFactor * 7, -Math.cos(state.yaw) * distance);
  const desired = bird.position.clone().add(offset);
  camera.position.lerp(desired, 1 - Math.pow(0.001, delta));
  const look = bird.position.clone().add(new THREE.Vector3(Math.sin(state.yaw) * 14, 1.5, Math.cos(state.yaw) * 14));
  camera.lookAt(look);
}

function updateWorld(delta, elapsed) {
  clouds.forEach(cloud => {
    cloud.position.x += cloud.userData.speed * delta;
    if (cloud.position.x > 250) cloud.position.x = -250;
  });
  entities.forEach(entity => {
    entity.update(elapsed);
    entity.group.rotation.z *= Math.pow(0.1, delta);
  });
}

function districtAt(x, z) {
  if (z < -95) return "Golden Gate";
  if (z < 8 && x < 60) return "Fisherman's Wharf";
  if (x > 86 && z < 100) return "Embarcadero";
  if (x > 28 && x < 94 && z >= 45 && z < 120) return "Chinatown";
  if (z > 135 && x < 20) return "Golden Gate Park";
  if (z < -20) return "San Francisco Bay";
  return "Nob Hill";
}

function drawMap() {
  const map = document.querySelector("#map");
  const context = map.getContext("2d");
  const width = map.width;
  const height = map.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#287e9c";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#e9ddb9";
  context.beginPath();
  context.moveTo(0, 46);
  context.lineTo(43, 42);
  context.lineTo(76, 48);
  context.lineTo(120, 40);
  context.lineTo(width, 48);
  context.lineTo(width, height);
  context.lineTo(0, height);
  context.fill();
  context.strokeStyle = "#1c2b2a";
  context.lineWidth = 2;
  context.stroke();
  context.strokeStyle = "#ef5b3f";
  context.lineWidth = 4;
  context.beginPath();
  context.moveTo(32, 6);
  context.lineTo(32, 49);
  context.stroke();
  const px = THREE.MathUtils.mapLinear(bird.position.x, -260, 270, 0, width);
  const py = THREE.MathUtils.mapLinear(bird.position.z, -260, 270, 0, height);
  context.save();
  context.translate(px, py);
  context.rotate(-state.yaw);
  context.fillStyle = "#f2bc3a";
  context.strokeStyle = "#1c2b2a";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(0, -7);
  context.lineTo(5, 6);
  context.lineTo(0, 3);
  context.lineTo(-5, 6);
  context.closePath();
  context.fill();
  context.stroke();
  context.restore();
  context.fillStyle = "#1c2b2a";
  context.font = "bold 9px Georgia";
  context.fillText("WHARF", 73, 39);
  context.fillText("CITY", 105, 79);
}

function updateReticle() {
  const center = new THREE.Vector2();
  let best = 1;
  entities.forEach(entity => {
    const projected = entity.group.position.clone().project(camera);
    if (projected.z < 1) {
      const distance = Math.hypot(projected.x - center.x, projected.y - center.y);
      best = Math.min(best, distance);
    }
  });
  document.querySelector("#reticle").classList.toggle("locked", best < 0.12);
}

function updateHud() {
  document.querySelector("#district").textContent = districtAt(bird.position.x, bird.position.z);
  document.querySelector("#score").textContent = String(state.score).padStart(3, "0");
  document.querySelector("#streak").textContent = state.streak > 1 ? `${state.streak}× streak` : "No streak yet";
  document.querySelector("#mode").textContent = state.mode;
  document.querySelector("#mode-icon").textContent = state.mode === "Flying" ? "↟" : state.mode === "Swimming" ? "≈" : "↝";
  const altitude = THREE.MathUtils.clamp((bird.position.y / 115) * 100, 4, 100);
  document.querySelector("#altitude").style.width = `${altitude}%`;
  const total = missionTotal();
  document.querySelector("#mission-count").textContent = `${total} / 6`;
  document.querySelector("#mission-progress").style.width = `${total / 6 * 100}%`;
  document.querySelector("#mission-text").textContent = `${state.mission.person}/3 people · ${state.mission.car}/2 cars · ${state.mission.animal}/1 animal`;
}

function setPaused(paused) {
  if (!state.started) return;
  state.paused = paused;
  document.querySelector("#pause-screen").classList.toggle("hidden", !paused);
}

function resetGame() {
  state.score = 0;
  state.streak = 0;
  state.mission = { person: 0, car: 0, animal: 0 };
  state.complete = false;
  state.paused = false;
  state.yaw = Math.PI;
  state.speed = 18;
  state.verticalSpeed = 0;
  bird.position.set(-88, 31, 12);
  document.querySelector("#pause-screen").classList.add("hidden");
  document.querySelector("#complete-screen").classList.add("hidden");
  updateHud();
}

document.addEventListener("keydown", event => {
  keys[event.code] = true;
  if (event.code === "KeyQ") dropPoop();
  if (event.code === "Escape") setPaused(!state.paused);
  if (["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.code)) event.preventDefault();
});

document.addEventListener("keyup", event => {
  keys[event.code] = false;
});

document.querySelector("#start-button").addEventListener("click", () => {
  state.started = true;
  document.querySelector("#start-screen").classList.add("hidden");
  document.querySelector("#hud").classList.remove("hidden");
  audioTone(520, 0.24, "triangle", 0.05);
  showToast("Find a target below!");
});

document.querySelector("#pause-button").addEventListener("click", () => setPaused(true));
document.querySelector("#resume-button").addEventListener("click", () => setPaused(false));
document.querySelector("#restart-button").addEventListener("click", resetGame);
document.querySelector("#continue-button").addEventListener("click", () => {
  state.paused = false;
  document.querySelector("#complete-screen").classList.add("hidden");
});
document.querySelector("#sound-button").addEventListener("click", event => {
  state.sound = !state.sound;
  event.currentTarget.textContent = state.sound ? "SFX" : "OFF";
  if (state.sound) audioTone(560, 0.12);
});

const joystick = document.querySelector("#joystick");
const joystickKnob = document.querySelector("#joystick-knob");

function updateJoystick(event) {
  const touch = event.touches?.[0] || event;
  const rect = joystick.getBoundingClientRect();
  const x = touch.clientX - rect.left - rect.width / 2;
  const y = touch.clientY - rect.top - rect.height / 2;
  const max = rect.width * 0.31;
  const length = Math.hypot(x, y) || 1;
  const clamped = Math.min(length, max);
  mobile.x = x / max;
  mobile.y = y / max;
  if (length > max) {
    mobile.x = x / length;
    mobile.y = y / length;
  }
  joystickKnob.style.transform = `translate(calc(-50% + ${mobile.x * clamped}px), calc(-50% + ${mobile.y * clamped}px))`;
}

function resetJoystick() {
  mobile.x = 0;
  mobile.y = 0;
  joystickKnob.style.transform = "translate(-50%, -50%)";
}

joystick.addEventListener("pointerdown", event => {
  joystick.setPointerCapture(event.pointerId);
  updateJoystick(event);
});
joystick.addEventListener("pointermove", event => {
  if (joystick.hasPointerCapture(event.pointerId)) updateJoystick(event);
});
joystick.addEventListener("pointerup", resetJoystick);
joystick.addEventListener("pointercancel", resetJoystick);

function bindHold(id, field) {
  const button = document.querySelector(id);
  button.addEventListener("pointerdown", event => {
    button.setPointerCapture(event.pointerId);
    mobile[field] = true;
  });
  button.addEventListener("pointerup", () => {
    mobile[field] = false;
  });
  button.addEventListener("pointercancel", () => {
    mobile[field] = false;
  });
}

bindHold("#flap-button", "flap");
bindHold("#dive-button", "dive");
document.querySelector("#poop-button").addEventListener("pointerdown", dropPoop);

window.addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setPixelRatio(Math.min(devicePixelRatio, innerWidth < 700 ? 1 : 1.1));
  renderer.setSize(innerWidth, innerHeight);
});

let hudAccumulator = 0;

function animate() {
  requestAnimationFrame(animate);
  const delta = Math.min(clock.getDelta(), 0.04);
  const elapsed = clock.elapsedTime;
  if (state.started && !state.paused) {
    updateBird(delta, elapsed);
    updateWorld(delta, elapsed);
    updatePoops(delta);
    updateCamera(delta);
    hudAccumulator += delta;
    if (hudAccumulator > 0.08) {
      updateHud();
      updateReticle();
      drawMap();
      hudAccumulator = 0;
    }
  } else if (!state.started) {
    water.rotation.z = Math.sin(elapsed * 0.1) * 0.005;
    bird.rotation.y += delta * 0.2;
    camera.position.set(-70, 70, 80);
    camera.lookAt(-20, 10, -30);
  }
  renderer.render(scene, camera);
}

updateHud();
animate();
