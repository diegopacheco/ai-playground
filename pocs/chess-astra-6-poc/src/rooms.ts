import * as THREE from 'three';
import { gold, ivory, stone, mesh, canvasTexture } from './parts';

export const backgrounds = { library: 'The Library', greatHall: 'The Great Hall', office: 'Dumbledore’s Office' } as const;
export type Background = keyof typeof backgrounds;
export type Room = { back: THREE.Group; side: THREE.Group; animate: (time: number) => void };

const oak = new THREE.MeshStandardMaterial({ color: '#593b28', roughness: 0.78 });
const darkOak = new THREE.MeshStandardMaterial({ color: '#33291f', roughness: 0.9 });
const sandstone = new THREE.MeshStandardMaterial({ color: '#c4b294', roughness: 0.95 });
const glass = new THREE.MeshBasicMaterial({ color: '#c6e3dc', side: THREE.DoubleSide });
const bookColors = ['#7d3431', '#32564c', '#3a4b67', '#95703a', '#5c3a57', '#aa7f54', '#4f6755'];
let spines: THREE.MeshStandardMaterial | null = null;

function spineMaterial() {
  if (spines) return spines;
  const spine = document.createElement('canvas');
  spine.width = 32;
  spine.height = 128;
  const paint = spine.getContext('2d')!;
  paint.fillStyle = '#ffffff';
  paint.fillRect(0, 0, 32, 128);
  paint.fillStyle = '#bcaa74';
  for (const y of [9, 15, 98, 104]) paint.fillRect(3, y, 26, 3);
  paint.strokeStyle = '#c5b88c';
  paint.strokeRect(7, 36, 18, 32);
  return spines = new THREE.MeshStandardMaterial({ map: new THREE.CanvasTexture(spine), roughness: 0.8 });
}

function bookcase(seed: number) {
  const shelf = new THREE.Group();
  shelf.add(mesh(new THREE.BoxGeometry(3.7, 7.1, 0.23), darkOak, 0, 3.55, -0.25));
  for (const x of [-1.9, 1.9]) {
    shelf.add(mesh(new THREE.BoxGeometry(0.2, 7.5, 0.75), oak, x, 3.65));
    shelf.add(mesh(new THREE.BoxGeometry(0.055, 7.3, 0.055), gold, x, 3.7, 0.4));
    shelf.add(mesh(new THREE.ConeGeometry(0.23, 0.55, 4), oak, x, 7.65));
  }
  for (let row = 0; row < 7; row++) {
    shelf.add(mesh(new THREE.BoxGeometry(3.9, 0.15, 0.83), oak, 0, row * 1.04 + 0.2));
    shelf.add(mesh(new THREE.BoxGeometry(3.9, 0.035, 0.035), gold, 0, row * 1.04 + 0.22, 0.44));
  }
  shelf.add(mesh(new THREE.BoxGeometry(4.1, 0.25, 0.9), oak, 0, 7.05));
  const books = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), spineMaterial(), 90);
  const transform = new THREE.Object3D();
  for (let i = 0; i < 90; i++) {
    const row = Math.floor(i / 15);
    const height = 0.55 + ((i * 17 + seed * 7) % 13) * 0.025;
    transform.position.set(-1.67 + i % 15 * 0.235, row * 1.04 + 0.28 + height / 2, 0.12);
    transform.scale.set(0.16 + i % 3 * 0.02, height, 0.43);
    transform.rotation.z = i % 13 === 0 ? 0.09 : 0;
    transform.updateMatrix();
    books.setMatrixAt(i, transform.matrix);
    books.setColorAt(i, new THREE.Color(bookColors[(i * 3 + seed) % bookColors.length]));
  }
  books.castShadow = true;
  books.receiveShadow = true;
  shelf.add(books);
  return shelf;
}

function archWindow(width: number, height: number, frame: number, pane: THREE.Material = glass) {
  const half = width / 2;
  const shape = new THREE.Shape();
  shape.moveTo(-half, 0);
  shape.lineTo(-half, height * 0.654);
  shape.quadraticCurveTo(-half, height * 0.859, 0, height);
  shape.quadraticCurveTo(half, height * 0.859, half, height * 0.654);
  shape.lineTo(half, 0);
  shape.closePath();
  const window = new THREE.Group();
  window.add(mesh(new THREE.ShapeGeometry(shape), pane));
  const outline = new THREE.CatmullRomCurve3(shape.getPoints(48).map(point => new THREE.Vector3(point.x, point.y, 0.11)), true);
  window.add(mesh(new THREE.TubeGeometry(outline, 96, frame, 8, true), stone));
  return window;
}

function banner(width: number, length: number) {
  const half = width / 2;
  const shape = new THREE.Shape();
  shape.moveTo(-half, 0);
  shape.lineTo(half, 0);
  shape.lineTo(half, width * 0.4 - length);
  shape.lineTo(0, -length);
  shape.lineTo(-half, width * 0.4 - length);
  shape.closePath();
  return new THREE.ShapeGeometry(shape);
}

function castleWalls(back: THREE.Group, side: THREE.Group) {
  back.add(mesh(new THREE.BoxGeometry(26, 12, 0.55), sandstone, 0, 5, -10));
  side.add(mesh(new THREE.BoxGeometry(0.5, 12, 18), sandstone, -11.6, 5, -1.1));
}

export function buildLibrary(): Room {
  const back = new THREE.Group();
  const side = new THREE.Group();
  const flames: THREE.Mesh[] = [];
  const firelight = new THREE.PointLight('#ffab4a', 24, 13, 2);
  castleWalls(back, side);
  for (let row = 0; row < 10; row++) {
    back.add(mesh(new THREE.BoxGeometry(26, 0.025, 0.035), stone, 0, row * 1.1 - 0.5, -9.7));
    for (let column = 0; column < 12; column++) {
      back.add(mesh(new THREE.BoxGeometry(0.025, 1.05, 0.035), stone, -12 + column * 2.2 + row % 2, row * 1.1, -9.7));
    }
  }
  for (let shelfIndex = 0; shelfIndex < 6; shelfIndex++) {
    const shelf = bookcase(shelfIndex);
    const sideShelf = shelfIndex >= 4;
    shelf.position.set(sideShelf ? -11 : [-8.7, -4.6, 4.6, 8.7][shelfIndex], -0.8, sideShelf ? (shelfIndex - 4) * 4.6 - 3.7 : -9.05);
    if (sideShelf) shelf.rotation.y = Math.PI / 2;
    (sideShelf ? side : back).add(shelf);
  }
  const window = archWindow(3.3, 7.8, 0.15);
  window.position.set(0, 0.7, -9.64);
  back.add(window);
  for (const x of [-0.83, 0, 0.83]) back.add(mesh(new THREE.BoxGeometry(0.085, 6.1, 0.12), gold, x, 3.85, -9.48));
  for (const y of [2, 3.8, 5.6]) back.add(mesh(new THREE.BoxGeometry(3.3, 0.07, 0.12), gold, 0, y, -9.48));
  const rose = mesh(new THREE.TorusGeometry(0.7, 0.055, 8, 48), gold, 0, 7.05, -9.43);
  back.add(rose);
  for (let i = 0; i < 8; i++) {
    const ray = mesh(new THREE.BoxGeometry(0.035, 1.4, 0.055), gold, 0, 7.05, -9.4);
    ray.rotation.z = i * Math.PI / 8;
    back.add(ray);
  }
  for (const x of [-11, -6.6, -2.3, 2.3, 6.6, 11]) {
    back.add(mesh(new THREE.CylinderGeometry(0.21, 0.34, 9, 12), stone, x, 3.6, -8.6));
    back.add(mesh(new THREE.BoxGeometry(0.8, 0.35, 0.8), sandstone, x, -0.65, -8.6));
    const curve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(x, 7.9, -8.6), new THREE.Vector3(x * 0.65, 10.8, -7), new THREE.Vector3(0, 12.1, -3.8));
    back.add(mesh(new THREE.TubeGeometry(curve, 24, 0.16, 8, false), stone));
  }
  for (const [index, x] of [-7, 7].entries()) {
    const cloth = new THREE.MeshStandardMaterial({ color: index ? '#31534b' : '#853e35', roughness: 1, side: THREE.DoubleSide });
    back.add(mesh(banner(1, 2.05), cloth, x, 6.8, -7.7));
    back.add(mesh(new THREE.BoxGeometry(1.25, 0.07, 0.07), gold, x, 6.82, -7.7));
    const crest = mesh(new THREE.OctahedronGeometry(0.23), gold, x, 6, -7.65);
    crest.scale.z = 0.2;
    back.add(crest);
  }
  const ladder = new THREE.Group();
  for (const x of [-0.38, 0.38]) ladder.add(mesh(new THREE.BoxGeometry(0.09, 5.1, 0.1), oak, x, 2.55));
  for (let i = 1; i < 10; i++) ladder.add(mesh(new THREE.BoxGeometry(0.85, 0.075, 0.1), oak, 0, i * 0.48));
  ladder.position.set(-7.6, -0.8, -7.6);
  ladder.rotation.x = -0.2;
  back.add(ladder);
  const desk = new THREE.Group();
  desk.add(mesh(new THREE.BoxGeometry(1.7, 0.13, 1.15), oak, 0, 0.5));
  for (const x of [-0.6, 0.6]) desk.add(mesh(new THREE.CylinderGeometry(0.07, 0.12, 1.3, 12), oak, x, -0.15));
  for (const direction of [-1, 1]) {
    const page = mesh(new THREE.BoxGeometry(0.52, 0.08, 0.72), ivory, direction * 0.25, 0.65);
    page.rotation.z = direction * 0.16;
    desk.add(page);
  }
  desk.position.set(-6.4, 0, -3.5);
  desk.rotation.y = 0.5;
  back.add(desk);
  const fireplace = new THREE.Group();
  const soot = new THREE.MeshStandardMaterial({ color: '#231c17', roughness: 1 });
  const ember = new THREE.MeshStandardMaterial({ color: '#562518', emissive: '#db4e12', emissiveIntensity: 1.5, roughness: 1 });
  fireplace.position.set(0, -0.8, -7.9);
  fireplace.add(mesh(new THREE.BoxGeometry(3.7, 0.2, 1.7), sandstone, 0, 0.1, -0.2));
  fireplace.add(mesh(new THREE.BoxGeometry(2.7, 2.05, 0.2), soot, 0, 1.05, -0.75));
  fireplace.add(mesh(new THREE.BoxGeometry(2.7, 0.08, 1.25), soot, 0, 0.23, -0.15));
  for (const x of [-1.5, 1.5]) {
    fireplace.add(mesh(new THREE.BoxGeometry(0.5, 2.5, 1.25), stone, x, 1.25, -0.25));
    for (let row = 0; row < 6; row++) fireplace.add(mesh(new THREE.BoxGeometry(0.55, 0.05, 1.3), sandstone, x, row * 0.4 + 0.15, -0.25));
  }
  fireplace.add(mesh(new THREE.BoxGeometry(3.85, 0.26, 1.6), sandstone, 0, 2.45, -0.25));
  fireplace.add(mesh(new THREE.BoxGeometry(3.55, 0.1, 1.4), gold, 0, 2.62, -0.25));
  fireplace.add(mesh(new THREE.BoxGeometry(3.2, 1.9, 0.6), stone, 0, 3.5, -0.6));
  const shield = mesh(new THREE.OctahedronGeometry(0.43), gold, 0, 3.45, -0.24);
  shield.scale.set(1, 1.2, 0.18);
  fireplace.add(shield);
  for (let i = 0; i < 4; i++) {
    const log = mesh(new THREE.CylinderGeometry(0.12, 0.17, 1.65, 10), ember, 0, 0.36 + i % 2 * 0.12, -0.4 + i * 0.17);
    log.rotation.z = Math.PI / 2;
    log.rotation.y = (i % 2 ? 1 : -1) * 0.3;
    fireplace.add(log);
  }
  for (let i = 0; i < 11; i++) {
    const flame = mesh(new THREE.LatheGeometry([[0, -0.15], [0.12, -0.11], [0.16, 0], [0.07, 0.12], [0, 0.24]].map(([x, y]) => new THREE.Vector2(x, y)), 12), new THREE.MeshBasicMaterial({ color: i % 2 ? '#ffc65a' : '#ff761e', transparent: true, opacity: 0.9, depthWrite: false }), Math.sin(i * 2.4) * 0.72, 0.8, -0.15 + Math.cos(i * 1.3) * 0.3);
    flame.scale.set(0.8, 2.2 + i % 3 * 0.5, 0.75);
    flame.userData.height = flame.scale.y;
    flame.castShadow = false;
    fireplace.add(flame);
    flames.push(flame);
    const core = mesh(new THREE.SphereGeometry(0.1, 10, 8), new THREE.MeshBasicMaterial({ color: '#ffe7a5' }), flame.position.x, 0.6, flame.position.z + 0.06);
    core.scale.y = 2;
    fireplace.add(core);
  }
  firelight.position.set(0, 1.1, 0.6);
  fireplace.add(firelight);
  back.add(fireplace);
  return {
    back,
    side,
    animate: time => {
      flames.forEach((flame, i) => {
        flame.scale.y = flame.userData.height * (0.85 + Math.sin(time * 0.008 + i * 2.3) * 0.2);
        flame.rotation.z = Math.sin(time * 0.006 + i) * 0.15;
      });
      firelight.intensity = 24 + Math.sin(time * 0.012) * 3 + Math.sin(time * 0.019) * 2;
    },
  };
}

function nightSky() {
  const texture = canvasTexture(128, 128, ctx => {
    ctx.fillStyle = '#1d2b4a';
    ctx.fillRect(0, 0, 128, 128);
    for (let i = 0; i < 70; i++) {
      ctx.fillStyle = `rgba(255, 244, 214, ${0.35 + Math.random() * 0.65})`;
      ctx.fillRect(Math.random() * 128, Math.random() * 128, 1 + Math.random() * 1.5, 1 + Math.random() * 1.5);
    }
  });
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(0.45, 0.45);
  return new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
}

function flagstones() {
  const texture = canvasTexture(256, 256, ctx => {
    ctx.fillStyle = '#6d6356';
    ctx.fillRect(0, 0, 256, 256);
    for (let row = 0; row < 4; row++) for (let column = 0; column < 4; column++) {
      const tone = 150 + Math.random() * 40;
      ctx.fillStyle = `rgb(${tone}, ${tone - 10}, ${tone - 24})`;
      ctx.fillRect(column * 64 + (row % 2) * 32 + 3, row * 64 + 3, 58, 58);
    }
  });
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(6, 3);
  return new THREE.MeshStandardMaterial({ map: texture, roughness: 0.95 });
}

export function buildGreatHall(): Room {
  const back = new THREE.Group();
  const side = new THREE.Group();
  castleWalls(back, side);
  const sky = nightSky();
  const floor = flagstones();
  const backFloor = mesh(new THREE.PlaneGeometry(26, 10), floor, 0, -0.88, -4.7);
  backFloor.rotation.x = -Math.PI / 2;
  const sideFloor = mesh(new THREE.PlaneGeometry(4, 18), floor, -9.35, -0.885, -1.1);
  sideFloor.rotation.x = -Math.PI / 2;
  back.add(backFloor);
  side.add(sideFloor);
  for (let row = 0; row < 10; row++) {
    back.add(mesh(new THREE.BoxGeometry(26, 0.025, 0.035), stone, 0, row * 1.1 - 0.5, -9.7));
    side.add(mesh(new THREE.BoxGeometry(0.035, 0.025, 18), stone, -11.33, row * 1.1 - 0.5, -1.1));
  }
  for (const x of [-7.5, -2.5, 2.5, 7.5]) {
    const window = archWindow(2.2, 6, 0.12, sky);
    window.position.set(x, 1.4, -9.66);
    back.add(window);
    back.add(mesh(new THREE.BoxGeometry(0.07, 5.8, 0.1), gold, x, 4.3, -9.5));
  }
  for (const z of [-5.5, 3]) {
    const window = archWindow(2.2, 6, 0.12, sky);
    window.position.set(-11.3, 1.4, z);
    window.rotation.y = Math.PI / 2;
    side.add(window);
  }
  ['#7a1f1f', '#1f4d3a', '#1f3b66', '#b58a2a'].forEach((color, i) => {
    const x = [-10, -5, 5, 10][i];
    back.add(mesh(new THREE.CylinderGeometry(0.25, 0.35, 10, 12), stone, x, 4.1, -9.3));
    back.add(mesh(new THREE.BoxGeometry(0.9, 0.35, 0.9), sandstone, x, -0.72, -9.3));
    back.add(mesh(banner(1.1, 2.9), new THREE.MeshStandardMaterial({ color, roughness: 1, side: THREE.DoubleSide }), x, 6.9, -8.88));
    back.add(mesh(new THREE.BoxGeometry(1.4, 0.07, 0.07), gold, x, 6.92, -8.85));
    const crest = mesh(new THREE.OctahedronGeometry(0.26), gold, x, 5.7, -8.8);
    crest.scale.z = 0.2;
    back.add(crest);
  });
  const shield = mesh(new THREE.OctahedronGeometry(0.5), gold, 0, 7.4, -9.55);
  shield.scale.set(1, 1.25, 0.18);
  back.add(shield);
  back.add(mesh(new THREE.BoxGeometry(18, 0.5, 2.8), sandstone, 0, -0.65, -8.4));
  back.add(mesh(new THREE.BoxGeometry(18.05, 0.05, 0.05), gold, 0, -0.4, -7));
  const highTable = new THREE.Group();
  highTable.position.set(0, -0.4, -8.2);
  highTable.add(mesh(new THREE.BoxGeometry(13, 0.12, 1.1), oak, 0, 0.9));
  highTable.add(mesh(new THREE.BoxGeometry(13.05, 0.02, 0.45), gold, 0, 0.97));
  highTable.add(mesh(new THREE.BoxGeometry(12.8, 0.8, 0.08), darkOak, 0, 0.45, 0.5));
  for (const x of [-6, -3, 0, 3, 6]) highTable.add(mesh(new THREE.BoxGeometry(0.7, x ? 1.9 : 2.6, 0.12), x ? darkOak : gold, x, x ? 0.95 : 1.3, -1));
  for (let i = 0; i < 10; i++) highTable.add(mesh(new THREE.CylinderGeometry(0.07, 0.04, 0.22, 10), gold, -5.4 + i * 1.2, 1.07, 0.1));
  back.add(highTable);
  const houseTable = new THREE.Group();
  houseTable.position.set(-9.9, -0.9, 0);
  houseTable.add(mesh(new THREE.BoxGeometry(1.2, 0.1, 13), oak, 0, 0.8));
  for (const z of [-5.8, 0, 5.8]) {
    houseTable.add(mesh(new THREE.BoxGeometry(0.9, 0.75, 0.12), darkOak, 0, 0.38, z));
    for (const x of [-0.9, 0.9]) houseTable.add(mesh(new THREE.BoxGeometry(0.3, 0.42, 0.1), darkOak, x, 0.21, z));
  }
  for (const x of [-0.9, 0.9]) houseTable.add(mesh(new THREE.BoxGeometry(0.35, 0.08, 13), oak, x, 0.45));
  for (let i = 0; i < 9; i++) {
    for (const x of [-0.3, 0.3]) houseTable.add(mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.02, 16), gold, x, 0.86, -5.6 + i * 1.4));
    houseTable.add(mesh(new THREE.CylinderGeometry(0.06, 0.035, 0.2, 10), gold, 0, 0.95, -5.6 + i * 1.4));
  }
  side.add(houseTable);
  const wax = new THREE.MeshStandardMaterial({ color: '#f3ead7', roughness: 0.6 });
  const glow = new THREE.MeshBasicMaterial({ color: '#ffd27a' });
  const halos = new THREE.MeshBasicMaterial({ color: '#ffb84a', transparent: true, opacity: 0.3, depthWrite: false, blending: THREE.AdditiveBlending });
  const candles: THREE.Group[] = [];
  for (let i = 0; i < 48; i++) {
    const nearSide = i % 3 === 0;
    const a = Math.sin(i * 12.9898) * 0.5 + 0.5;
    const b = Math.sin(i * 78.233) * 0.5 + 0.5;
    const candle = new THREE.Group();
    candle.position.set(nearSide ? -11 + a * 5 : -11 + a * 22, 4 + (i * 0.37) % 1 * 5, nearSide ? -4 + b * 9 : -9.2 + b * 4);
    candle.userData.y = candle.position.y;
    const body = mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.5, 8), wax);
    const flame = mesh(new THREE.ConeGeometry(0.08, 0.26, 8), glow, 0, 0.38);
    const halo = mesh(new THREE.SphereGeometry(0.2, 12, 8), halos, 0, 0.36);
    body.castShadow = flame.castShadow = halo.castShadow = false;
    candle.add(body, flame, halo);
    (nearSide ? side : back).add(candle);
    candles.push(candle);
  }
  const candlelight = new THREE.PointLight('#ffcf7a', 30, 18, 2);
  candlelight.position.set(0, 6, -6);
  back.add(candlelight);
  return {
    back,
    side,
    animate: time => candles.forEach((candle, i) => {
      candle.position.y = candle.userData.y + Math.sin(time * 0.0012 + i) * 0.12;
      candle.children[1].scale.y = 1 + Math.sin(time * 0.02 + i * 3) * 0.25;
    }),
  };
}

function portrait(index: number) {
  const robes = ['#5b2a6e', '#2f4f7a', '#7a2f2f', '#2f6a4f', '#8a6a2a'];
  const texture = canvasTexture(64, 80, ctx => {
    ctx.fillStyle = ['#2a2320', '#1f2a2a', '#2b2433'][index % 3];
    ctx.fillRect(0, 0, 64, 80);
    ctx.fillStyle = robes[index % robes.length];
    ctx.beginPath();
    ctx.moveTo(8, 80);
    ctx.lineTo(20, 47);
    ctx.lineTo(44, 47);
    ctx.lineTo(56, 80);
    ctx.fill();
    ctx.fillStyle = '#dfbd97';
    ctx.beginPath();
    ctx.ellipse(32, 36, 9, 11, 0, 0, Math.PI * 2);
    ctx.fill();
    if (index % 2 === 0) {
      ctx.fillStyle = '#dcd8cf';
      ctx.beginPath();
      ctx.moveTo(23, 40);
      ctx.lineTo(41, 40);
      ctx.lineTo(32, 64);
      ctx.fill();
    }
    if (index % 3 === 1) {
      ctx.fillStyle = robes[(index + 2) % robes.length];
      ctx.beginPath();
      ctx.moveTo(19, 28);
      ctx.lineTo(45, 28);
      ctx.lineTo(34, 3);
      ctx.fill();
    }
  });
  const frame = new THREE.Group();
  frame.add(mesh(new THREE.BoxGeometry(1, 1.25, 0.08), gold));
  frame.add(mesh(new THREE.PlaneGeometry(0.82, 1.05), new THREE.MeshStandardMaterial({ map: texture, roughness: 0.9 }), 0, 0, 0.05));
  return frame;
}

function onWall<T extends THREE.Object3D>(object: T, radius: number, degrees: number, y: number) {
  const angle = degrees * Math.PI / 180;
  object.position.set(radius * Math.sin(angle), y, radius * Math.cos(angle));
  object.rotation.y = angle + Math.PI;
  return object;
}

function curvedWall(group: THREE.Group, from: number, to: number, material: THREE.Material, wainscot: THREE.Material, trim: THREE.Material, rug: THREE.Material) {
  const start = from * Math.PI / 180;
  const length = (to - from) * Math.PI / 180;
  const band = (radius: number, height: number, y: number, surface: THREE.Material) => group.add(mesh(new THREE.CylinderGeometry(radius, radius, height, 48, 1, true, start, length), surface, 0, y));
  band(10.4, 12, 5.1, material);
  band(10.35, 2.4, 0.3, wainscot);
  band(10.3, 0.1, 1.5, trim);
  const carpet = mesh(new THREE.RingGeometry(8.15, 10.35, 48, 1, start - Math.PI / 2, length), rug, 0, -0.88);
  carpet.rotation.x = -Math.PI / 2;
  group.add(carpet);
}

export function buildOffice(): Room {
  const back = new THREE.Group();
  const side = new THREE.Group();
  const plaster = new THREE.MeshStandardMaterial({ color: '#8f604a', roughness: 0.92, side: THREE.BackSide });
  const panel = new THREE.MeshStandardMaterial({ color: '#3a2a20', roughness: 0.85, side: THREE.BackSide });
  const trim = new THREE.MeshStandardMaterial({ color: '#b39051', metalness: 0.72, roughness: 0.3, side: THREE.BackSide });
  const rug = new THREE.MeshStandardMaterial({ color: '#5c2026', roughness: 1 });
  curvedWall(back, 126, 234, plaster, panel, trim, rug);
  curvedWall(side, 234, 300, plaster, panel, trim, rug);
  const shelves = [140, 220, 270].map(degrees => onWall(bookcase(degrees), 9.85, degrees, -0.8));
  back.add(shelves[0], shelves[1]);
  side.add(shelves[2]);
  let index = 0;
  for (const degrees of [158, 169, 180, 191, 202]) for (const y of [3.6, 5.2, 6.8]) back.add(onWall(portrait(index++), 10.25, degrees + (y === 5.2 ? 3 : 0), y));
  for (const degrees of [132, 144, 156, 168, 180, 192, 204, 216, 228]) back.add(onWall(portrait(index++), 10.25, degrees, 8.9));
  for (const degrees of [244, 254, 286, 295]) for (const y of [3.6, 5.4]) side.add(onWall(portrait(index++), 10.25, degrees, y));
  for (const degrees of [246, 258, 270, 282, 294]) side.add(onWall(portrait(index++), 10.25, degrees, 8.9));
  const felt = new THREE.MeshStandardMaterial({ color: '#5b4330', roughness: 1 });
  const hat = new THREE.Group();
  hat.add(mesh(new THREE.CylinderGeometry(0.45, 0.45, 0.04, 20), felt, 0, 0.02));
  hat.add(mesh(new THREE.ConeGeometry(0.26, 0.75, 16), felt, 0, 0.39));
  const tip = mesh(new THREE.ConeGeometry(0.09, 0.3, 10), felt, 0.1, 0.8);
  tip.rotation.z = -0.9;
  hat.add(tip);
  hat.position.set(0, 7.18, 0.1);
  shelves[0].add(hat);
  const velvet = new THREE.MeshStandardMaterial({ color: '#6a1f2a', roughness: 0.8 });
  const silver = new THREE.MeshStandardMaterial({ color: '#cfd3d8', metalness: 0.5, roughness: 0.3 });
  const desk = new THREE.Group();
  desk.position.set(0, -0.9, -8.5);
  desk.add(mesh(new THREE.BoxGeometry(3.4, 0.14, 1.5), oak, 0, 1));
  for (const x of [-1.4, 1.4]) desk.add(mesh(new THREE.BoxGeometry(0.5, 0.93, 1.3), darkOak, x, 0.47));
  desk.add(mesh(new THREE.BoxGeometry(2.3, 0.7, 0.08), darkOak, 0, 0.6, 0.6));
  for (const x of [-1.55, 1.55]) for (const z of [-0.62, 0.62]) desk.add(mesh(new THREE.SphereGeometry(0.1, 12, 8), gold, x, 0.08, z));
  desk.add(mesh(new THREE.BoxGeometry(1.1, 2.9, 0.18), velvet, 0, 1.45, -1.25));
  desk.add(mesh(new THREE.BoxGeometry(1.2, 0.12, 0.25), gold, 0, 2.95, -1.25));
  desk.add(mesh(new THREE.BoxGeometry(0.6, 0.12, 0.45), velvet, -0.9, 1.13, 0.1));
  const tome = mesh(new THREE.BoxGeometry(0.5, 0.1, 0.4), darkOak, -0.88, 1.24, 0.08);
  tome.rotation.y = 0.3;
  desk.add(tome);
  const rings: THREE.Mesh[] = [];
  for (const x of [0.5, 1.15]) {
    const instrument = new THREE.Group();
    instrument.position.set(x, 1.07, 0.1);
    instrument.add(mesh(new THREE.CylinderGeometry(0.02, 0.08, 0.35, 8), silver, 0, 0.17));
    instrument.add(mesh(new THREE.SphereGeometry(0.05, 12, 8), gold, 0, 0.45));
    const outer = mesh(new THREE.TorusGeometry(0.16, 0.012, 6, 32), silver, 0, 0.45);
    const inner = mesh(new THREE.TorusGeometry(0.11, 0.012, 6, 32), gold, 0, 0.45);
    inner.rotation.x = Math.PI / 2;
    instrument.add(outer, inner);
    rings.push(outer, inner);
    desk.add(instrument);
  }
  back.add(desk);
  const plume = new THREE.MeshStandardMaterial({ color: '#b3261e', emissive: '#4a0b05', roughness: 0.6 });
  const flare = new THREE.MeshStandardMaterial({ color: '#e0892a', emissive: '#5a2800', roughness: 0.6 });
  const perch = new THREE.Group();
  perch.position.set(-3.7, -0.9, -8.3);
  perch.add(mesh(new THREE.CylinderGeometry(0.35, 0.45, 0.12, 16), gold, 0, 0.06));
  perch.add(mesh(new THREE.CylinderGeometry(0.04, 0.06, 2.6, 8), gold, 0, 1.3));
  perch.add(mesh(new THREE.BoxGeometry(0.9, 0.05, 0.05), gold, 0, 2.6));
  const phoenix = new THREE.Group();
  phoenix.position.set(0, 2.88, 0);
  const body = mesh(new THREE.SphereGeometry(0.22, 16, 12), plume);
  body.scale.set(0.8, 1.15, 0.9);
  phoenix.add(body, mesh(new THREE.SphereGeometry(0.12, 14, 10), plume, 0, 0.32, 0.06));
  const beak = mesh(new THREE.ConeGeometry(0.035, 0.12, 8), gold, 0, 0.3, 0.2);
  beak.rotation.x = Math.PI / 2;
  phoenix.add(beak, mesh(new THREE.ConeGeometry(0.03, 0.16, 8), flare, 0, 0.47, 0.02));
  for (const direction of [-1, 1]) {
    const wing = mesh(new THREE.SphereGeometry(0.2, 12, 8), flare, direction * 0.18, 0.02, -0.02);
    wing.scale.set(0.3, 1, 0.8);
    wing.rotation.z = direction * 0.25;
    phoenix.add(wing);
  }
  for (let i = 0; i < 3; i++) {
    const feather = mesh(new THREE.ConeGeometry(0.05, 0.9, 8), i === 1 ? plume : flare, -0.08 + i * 0.08, -0.6, -0.14);
    feather.rotation.x = Math.PI - 0.25;
    phoenix.add(feather);
  }
  perch.add(phoenix);
  back.add(perch);
  const pensieve = new THREE.Group();
  pensieve.position.set(3.7, -0.9, -8.2);
  pensieve.add(mesh(new THREE.CylinderGeometry(0.25, 0.4, 0.9, 12), stone, 0, 0.45));
  pensieve.add(mesh(new THREE.CylinderGeometry(0.75, 0.45, 0.35, 24), stone, 0, 1.07));
  const rim = mesh(new THREE.TorusGeometry(0.72, 0.04, 8, 40), gold, 0, 1.25);
  rim.rotation.x = Math.PI / 2;
  const memory = new THREE.MeshBasicMaterial({ color: '#bfe3ff', transparent: true, opacity: 0.9 });
  const pool = mesh(new THREE.CircleGeometry(0.68, 32), memory, 0, 1.252);
  pool.rotation.x = -Math.PI / 2;
  const wisps = [0, 1].map(i => {
    const wisp = mesh(new THREE.TorusGeometry(0.3 + i * 0.12, 0.01, 6, 40), new THREE.MeshBasicMaterial({ color: '#eef8ff', transparent: true, opacity: 0.6 }), 0, 1.45 + i * 0.12);
    wisp.castShadow = false;
    return wisp;
  });
  const glow = new THREE.PointLight('#a8d4ff', 12, 9, 2);
  glow.position.set(0, 1.9, 0.3);
  pensieve.add(rim, pool, ...wisps, glow);
  back.add(pensieve);
  return {
    back,
    side,
    animate: time => {
      rings.forEach((ring, i) => { ring.rotation.y = time * (i % 2 ? 0.0021 : -0.0014); });
      phoenix.rotation.y = Math.sin(time * 0.0007) * 0.35;
      memory.opacity = 0.75 + Math.sin(time * 0.002) * 0.15;
      wisps.forEach((wisp, i) => {
        wisp.rotation.x = Math.PI / 2 + Math.sin(time * 0.001 + i) * 0.35;
        wisp.rotation.y = time * 0.0009 * (i ? 1 : -1);
      });
    },
  };
}
