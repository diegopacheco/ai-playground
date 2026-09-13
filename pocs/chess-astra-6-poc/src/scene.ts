import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Chess, type Square, type PieceSymbol, type Move } from 'chess.js';

const gold = new THREE.MeshStandardMaterial({ color: '#b39051', metalness: 0.72, roughness: 0.3 });
const ivory = new THREE.MeshStandardMaterial({ color: '#f9ecd2', metalness: 0.15, roughness: 0.3 });
const jade = new THREE.MeshStandardMaterial({ color: '#31574d', metalness: 0.4, roughness: 0.3 });
export const pieceColors = { white: '#f9ecd2', green: '#31574d', brown: '#75472d', black: '#20272a' } as const;
export type PieceColor = keyof typeof pieceColors;

const stone = new THREE.MeshStandardMaterial({ color: '#d5cbb9', roughness: 0.9 });

function mesh(geometry: THREE.BufferGeometry, material: THREE.Material, x = 0, y = 0, z = 0) {
  const object = new THREE.Mesh(geometry, material);
  object.position.set(x, y, z);
  object.castShadow = true;
  object.receiveShadow = true;
  return object;
}

function cylinder(group: THREE.Group, top: number, bottom: number, height: number, y: number, material: THREE.Material) {
  group.add(mesh(new THREE.CylinderGeometry(top, bottom, height, 32), material, 0, y));
}

function piece(type: PieceSymbol, material: THREE.MeshStandardMaterial): THREE.Group {
  const group = new THREE.Group();
  cylinder(group, 0.31, 0.34, 0.1, 0.05, material);
  cylinder(group, 0.3, 0.31, 0.045, 0.12, gold);
  cylinder(group, 0.24, 0.29, 0.08, 0.18, material);
  const height = type === 'p' ? 0.5 : type === 'k' || type === 'q' ? 0.91 : 0.72;
  const profile = [[0.24, 0.21], [0.21, 0.27], [0.14, 0.34], [0.105, height - 0.08], [0.19, height], [0.19, height + 0.045]];
  group.add(mesh(new THREE.LatheGeometry(profile.map(([x, y]) => new THREE.Vector2(x, y)), 32), material));
  cylinder(group, 0.2, 0.2, 0.035, height + 0.045, gold);
  if (type === 'p') {
    group.add(mesh(new THREE.SphereGeometry(0.18, 24, 16), material, 0, height + 0.2));
  } else if (type === 'r') {
    cylinder(group, 0.25, 0.2, 0.22, height + 0.16, material);
    for (let i = 0; i < 6; i++) {
      const angle = i * Math.PI / 3;
      const merlon = mesh(new THREE.BoxGeometry(0.13, 0.14, 0.13), material, Math.cos(angle) * 0.19, height + 0.31, Math.sin(angle) * 0.19);
      merlon.rotation.y = -angle;
      group.add(merlon);
    }
  } else if (type === 'n') {
    const shape = new THREE.Shape();
    shape.moveTo(-0.2, 0);
    shape.lineTo(-0.22, 0.29);
    shape.lineTo(-0.13, 0.52);
    shape.lineTo(-0.16, 0.69);
    shape.lineTo(-0.02, 0.62);
    shape.lineTo(0.12, 0.58);
    shape.lineTo(0.28, 0.38);
    shape.lineTo(0.28, 0.25);
    shape.lineTo(0.1, 0.24);
    shape.lineTo(0.01, 0.33);
    shape.lineTo(0.04, 0.08);
    shape.lineTo(0.2, 0);
    shape.closePath();
    const head = mesh(new THREE.ExtrudeGeometry(shape, { depth: 0.19, bevelEnabled: true, bevelSegments: 2, steps: 1, bevelSize: 0.045, bevelThickness: 0.035 }), material, 0, height + 0.04, -0.095);
    group.add(head);
    group.add(mesh(new THREE.SphereGeometry(0.032, 12, 8), gold, 0.045, height + 0.5, 0.145));
    group.add(mesh(new THREE.SphereGeometry(0.032, 12, 8), gold, 0.045, height + 0.5, -0.145));
  } else if (type === 'b') {
    group.add(mesh(new THREE.SphereGeometry(0.19, 24, 16), material, 0, height + 0.22));
    group.add(mesh(new THREE.ConeGeometry(0.16, 0.3, 24), material, 0, height + 0.4));
    group.add(mesh(new THREE.SphereGeometry(0.06, 16, 12), gold, 0, height + 0.57));
    const slash = mesh(new THREE.BoxGeometry(0.028, 0.17, 0.03), gold, 0, height + 0.36, 0.14);
    slash.rotation.z = -0.45;
    group.add(slash);
  } else {
    cylinder(group, 0.26, 0.15, 0.22, height + 0.2, material);
    for (let i = 0; i < 7; i++) {
      const angle = i * Math.PI * 2 / 7;
      group.add(mesh(new THREE.ConeGeometry(0.055, 0.19, 8), gold, Math.cos(angle) * 0.22, height + 0.38, Math.sin(angle) * 0.22));
    }
    group.add(mesh(new THREE.SphereGeometry(0.1, 16, 12), material, 0, height + 0.36));
    if (type === 'k') {
      group.add(mesh(new THREE.BoxGeometry(0.065, 0.32, 0.065), gold, 0, height + 0.58));
      group.add(mesh(new THREE.BoxGeometry(0.22, 0.065, 0.065), gold, 0, height + 0.61));
    }
  }
  return group;
}

export class ChessScene {
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(37, 1, 0.1, 100);
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private pieces = new THREE.Group();
  private playerMaterial = ivory.clone();
  private cpuMaterial = jade.clone();
  private library = new THREE.Group();
  private sideLibrary = new THREE.Group();
  private markers = new THREE.Group();
  private tiles: THREE.Mesh[] = [];
  private flames: THREE.Mesh[] = [];
  private firelight = new THREE.PointLight('#ffab4a', 24, 13, 2);
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();
  private animations: { object: THREE.Object3D; from: THREE.Vector3; to: THREE.Vector3; start: number }[] = [];
  private reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  private dust: THREE.Points;
  private captures: { group: THREE.Group; victim: THREE.Object3D | null; shards: THREE.Mesh[]; ring: THREE.Mesh; start: number }[] = [];
  private fallingKing: { object: THREE.Object3D; start: number; shattered: boolean } | null = null;
  private overhead = false;
  private flipped = false;

  constructor(private container: HTMLElement, onSquare: (square: Square) => void) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.setClearColor('#eee9df', 0);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;
    this.renderer.domElement.setAttribute('aria-label', 'Interactive 3D chessboard. Select a piece, then a highlighted square. You can also use the move input below.');
    this.renderer.domElement.setAttribute('role', 'img');
    container.prepend(this.renderer.domElement);
    this.scene.fog = new THREE.Fog('#e4dccd', 32, 65);
    this.scene.add(new THREE.HemisphereLight('#fff6df', '#758478', 2.4));
    const sun = new THREE.DirectionalLight('#fff1d5', 3);
    sun.position.set(-5, 12, 7);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -9, right: 9, top: 9, bottom: -9 });
    sun.shadow.bias = -0.0005;
    this.scene.add(sun);
    this.scene.add(mesh(new THREE.BoxGeometry(9.25, 0.25, 9.25), gold, 0, -0.34));
    this.scene.add(mesh(new THREE.BoxGeometry(9.05, 0.43, 9.05), jade, 0, -0.13));
    this.scene.add(mesh(new THREE.BoxGeometry(8.7, 0.09, 8.7), gold, 0, 0.1));
    const lightTile = new THREE.MeshStandardMaterial({ color: '#e9ddc5', roughness: 0.58, metalness: 0.08 });
    const darkTile = new THREE.MeshStandardMaterial({ color: '#648477', roughness: 0.5, metalness: 0.14 });
    for (let rank = 0; rank < 8; rank++) {
      for (let file = 0; file < 8; file++) {
        const square = `${String.fromCharCode(97 + file)}${rank + 1}`;
        const tile = mesh(new THREE.BoxGeometry(0.994, 0.08, 0.994), (rank + file) % 2 ? lightTile : darkTile, file - 3.5, 0.17, 3.5 - rank);
        tile.userData.square = square;
        this.tiles.push(tile);
        this.scene.add(tile);
      }
    }
    for (let i = 0; i < 8; i++) {
      this.label(String.fromCharCode(97 + i), i - 3.5, 4.26);
      this.label(String(i + 1), -4.27, 3.5 - i);
    }
    this.scene.add(this.pieces, this.markers);
    this.buildHall();
    const points = new Float32Array(180 * 3);
    for (let i = 0; i < points.length; i += 3) {
      points[i] = Math.sin(i * 23.1) * 8;
      points[i + 1] = 0.5 + ((i * 0.73) % 5);
      points[i + 2] = Math.cos(i * 7.2) * 8;
    }
    this.dust = new THREE.Points(new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(points, 3)), new THREE.PointsMaterial({ color: '#e4bc68', size: 0.035, transparent: true, opacity: 0.7 }));
    this.scene.add(this.dust);
    this.camera.position.set(10.3, 13.4, 15.8);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.1, 0);
    this.controls.enableDamping = true;
    this.controls.enablePan = false;
    this.controls.minDistance = 12;
    this.controls.maxDistance = 27;
    this.controls.minPolarAngle = 0.08;
    this.controls.maxPolarAngle = Math.PI / 2.3;
    let down = { x: 0, y: 0 };
    this.renderer.domElement.addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY }; });
    this.renderer.domElement.addEventListener('pointerup', e => {
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6) return;
      const rect = this.renderer.domElement.getBoundingClientRect();
      this.pointer.set((e.clientX - rect.left) / rect.width * 2 - 1, -(e.clientY - rect.top) / rect.height * 2 + 1);
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const hits = this.raycaster.intersectObjects([...this.pieces.children, ...this.tiles], true);
      for (const hit of hits) {
        let object: THREE.Object3D | null = hit.object;
        while (object && !object.userData.square) object = object.parent;
        if (object?.userData.square) { onSquare(object.userData.square); return; }
      }
    });
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.renderer.setAnimationLoop(time => this.frame(time));
  }

  private label(text: string, x: number, z: number) {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 64;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#f3dab0';
    ctx.font = '40px Georgia';
    ctx.textAlign = 'center';
    ctx.fillText(text, 32, 46);
    const label = mesh(new THREE.PlaneGeometry(0.28, 0.28), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true }), x, 0.22, z);
    label.rotation.x = -Math.PI / 2;
    this.scene.add(label);
  }

  private buildHall() {
    const floor = mesh(new THREE.CylinderGeometry(7.8, 8.1, 0.35, 64), stone, 0, -0.69);
    this.scene.add(floor);
    const floorRing = mesh(new THREE.TorusGeometry(7.65, 0.025, 8, 100), gold, 0, -0.49);
    floorRing.rotation.x = Math.PI / 2;
    this.scene.add(floorRing);
    const ground = mesh(new THREE.PlaneGeometry(150, 150), new THREE.MeshStandardMaterial({ color: '#e9e3d7', roughness: 1 }), 0, -0.9);
    ground.rotation.x = -Math.PI / 2;
    this.scene.add(ground);
    this.scene.add(this.library, this.sideLibrary);
    const oak = new THREE.MeshStandardMaterial({ color: '#593b28', roughness: 0.78 });
    const darkOak = new THREE.MeshStandardMaterial({ color: '#33291f', roughness: 0.9 });
    const sandstone = new THREE.MeshStandardMaterial({ color: '#c4b294', roughness: 0.95 });
    const glass = new THREE.MeshBasicMaterial({ color: '#c6e3dc', side: THREE.DoubleSide });
    const wall = mesh(new THREE.BoxGeometry(26, 12, 0.55), sandstone, 0, 5, -10);
    this.library.add(wall);
    const leftWall = mesh(new THREE.BoxGeometry(0.5, 12, 18), sandstone, -11.6, 5, -1.1);
    this.sideLibrary.add(leftWall);
    for (let row = 0; row < 10; row++) {
      this.library.add(mesh(new THREE.BoxGeometry(26, 0.025, 0.035), stone, 0, row * 1.1 - 0.5, -9.7));
      for (let column = 0; column < 12; column++) {
        this.library.add(mesh(new THREE.BoxGeometry(0.025, 1.05, 0.035), stone, -12 + column * 2.2 + row % 2, row * 1.1, -9.7));
      }
    }
    const bookColors = ['#7d3431', '#32564c', '#3a4b67', '#95703a', '#5c3a57', '#aa7f54', '#4f6755'];
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
    const bookMaterial = new THREE.MeshStandardMaterial({ map: new THREE.CanvasTexture(spine), roughness: 0.8 });
    for (let shelfIndex = 0; shelfIndex < 6; shelfIndex++) {
      const shelf = new THREE.Group();
      const sideShelf = shelfIndex >= 4;
      shelf.position.set(sideShelf ? -11 : [-8.7, -4.6, 4.6, 8.7][shelfIndex], -0.8, sideShelf ? (shelfIndex - 4) * 4.6 - 3.7 : -9.05);
      if (sideShelf) shelf.rotation.y = Math.PI / 2;
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
      const books = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), bookMaterial, 90);
      const transform = new THREE.Object3D();
      for (let i = 0; i < 90; i++) {
        const row = Math.floor(i / 15);
        const height = 0.55 + ((i * 17 + shelfIndex * 7) % 13) * 0.025;
        transform.position.set(-1.67 + i % 15 * 0.235, row * 1.04 + 0.28 + height / 2, 0.12);
        transform.scale.set(0.16 + i % 3 * 0.02, height, 0.43);
        transform.rotation.z = i % 13 === 0 ? 0.09 : 0;
        transform.updateMatrix();
        books.setMatrixAt(i, transform.matrix);
        books.setColorAt(i, new THREE.Color(bookColors[(i * 3 + shelfIndex) % bookColors.length]));
      }
      books.castShadow = true;
      books.receiveShadow = true;
      shelf.add(books);
      (sideShelf ? this.sideLibrary : this.library).add(shelf);
    }
    const windowShape = new THREE.Shape();
    windowShape.moveTo(-1.65, 0);
    windowShape.lineTo(-1.65, 5.1);
    windowShape.quadraticCurveTo(-1.65, 6.7, 0, 7.8);
    windowShape.quadraticCurveTo(1.65, 6.7, 1.65, 5.1);
    windowShape.lineTo(1.65, 0);
    windowShape.closePath();
    const windowPane = mesh(new THREE.ShapeGeometry(windowShape), glass, 0, 0.7, -9.64);
    this.library.add(windowPane);
    const outline = new THREE.CatmullRomCurve3(windowShape.getPoints(48).map(point => new THREE.Vector3(point.x, point.y + 0.7, -9.53)), true);
    this.library.add(mesh(new THREE.TubeGeometry(outline, 96, 0.15, 8, true), stone));
    for (const x of [-0.83, 0, 0.83]) this.library.add(mesh(new THREE.BoxGeometry(0.085, 6.1, 0.12), gold, x, 3.85, -9.48));
    for (const y of [2, 3.8, 5.6]) this.library.add(mesh(new THREE.BoxGeometry(3.3, 0.07, 0.12), gold, 0, y, -9.48));
    const rose = mesh(new THREE.TorusGeometry(0.7, 0.055, 8, 48), gold, 0, 7.05, -9.43);
    this.library.add(rose);
    for (let i = 0; i < 8; i++) {
      const ray = mesh(new THREE.BoxGeometry(0.035, 1.4, 0.055), gold, 0, 7.05, -9.4);
      ray.rotation.z = i * Math.PI / 8;
      this.library.add(ray);
    }
    for (const x of [-11, -6.6, -2.3, 2.3, 6.6, 11]) {
      this.library.add(mesh(new THREE.CylinderGeometry(0.21, 0.34, 9, 12), stone, x, 3.6, -8.6));
      this.library.add(mesh(new THREE.BoxGeometry(0.8, 0.35, 0.8), sandstone, x, -0.65, -8.6));
      const curve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(x, 7.9, -8.6), new THREE.Vector3(x * 0.65, 10.8, -7), new THREE.Vector3(0, 12.1, -3.8));
      this.library.add(mesh(new THREE.TubeGeometry(curve, 24, 0.16, 8, false), stone));
    }
    for (const [index, x] of [-7, 7].entries()) {
      const bannerShape = new THREE.Shape();
      bannerShape.moveTo(-0.5, 0);
      bannerShape.lineTo(0.5, 0);
      bannerShape.lineTo(0.5, -1.65);
      bannerShape.lineTo(0, -2.05);
      bannerShape.lineTo(-0.5, -1.65);
      bannerShape.closePath();
      const cloth = new THREE.MeshStandardMaterial({ color: index ? '#31534b' : '#853e35', roughness: 1, side: THREE.DoubleSide });
      this.library.add(mesh(new THREE.ShapeGeometry(bannerShape), cloth, x, 6.8, -7.7));
      this.library.add(mesh(new THREE.BoxGeometry(1.25, 0.07, 0.07), gold, x, 6.82, -7.7));
      const crest = mesh(new THREE.OctahedronGeometry(0.23), gold, x, 6, -7.65);
      crest.scale.z = 0.2;
      this.library.add(crest);
    }
    const ladder = new THREE.Group();
    for (const x of [-0.38, 0.38]) ladder.add(mesh(new THREE.BoxGeometry(0.09, 5.1, 0.1), oak, x, 2.55));
    for (let i = 1; i < 10; i++) ladder.add(mesh(new THREE.BoxGeometry(0.85, 0.075, 0.1), oak, 0, i * 0.48));
    ladder.position.set(-7.6, -0.8, -7.6);
    ladder.rotation.x = -0.2;
    this.library.add(ladder);
    const desk = new THREE.Group();
    desk.add(mesh(new THREE.BoxGeometry(1.7, 0.13, 1.15), oak, 0, 0.5));
    for (const x of [-0.6, 0.6]) desk.add(mesh(new THREE.CylinderGeometry(0.07, 0.12, 1.3, 12), oak, x, -0.15));
    for (const side of [-1, 1]) {
      const page = mesh(new THREE.BoxGeometry(0.52, 0.08, 0.72), ivory, side * 0.25, 0.65);
      page.rotation.z = side * 0.16;
      desk.add(page);
    }
    desk.position.set(-6.4, 0, -3.5);
    desk.rotation.y = 0.5;
    this.library.add(desk);
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
      this.flames.push(flame);
      const core = mesh(new THREE.SphereGeometry(0.1, 10, 8), new THREE.MeshBasicMaterial({ color: '#ffe7a5' }), flame.position.x, 0.6, flame.position.z + 0.06);
      core.scale.y = 2;
      fireplace.add(core);
    }
    this.firelight.position.set(0, 1.1, 0.6);
    fireplace.add(this.firelight);
    this.library.add(fireplace);
  }

  setPieceColor(color: PieceColor) {
    this.playerMaterial.color.set(pieceColors[color]);
    this.cpuMaterial.color.set(pieceColors[color === 'white' ? 'green' : 'white']);
  }

  sync(game: Chess, move?: Move) {
    if (!move) this.clearCaptures();
    this.fallingKing = null;
    if (move?.captured && !this.reducedMotion) {
      const square = move.isEnPassant() ? `${move.to[0]}${move.from[1]}` : move.to;
      const victim = this.pieces.children.find(object => object.userData.square === square);
      if (victim) this.capture(victim);
    }
    const old = new Map(this.pieces.children.map(p => [p.userData.square, p.position.clone()]));
    this.pieces.traverse(object => { if (object instanceof THREE.Mesh) object.geometry.dispose(); });
    this.pieces.clear();
    this.animations = [];
    for (const row of game.board()) for (const p of row) {
      if (!p) continue;
      const object = piece(p.type, p.color === 'w' ? this.playerMaterial : this.cpuMaterial);
      object.userData.square = p.square;
      object.position.set(p.square.charCodeAt(0) - 100.5, 0.22, 4.5 - Number(p.square[1]));
      if (move?.to === p.square && old.has(move.from) && !this.reducedMotion) {
        this.animations.push({ object, from: old.get(move.from)!, to: object.position.clone(), start: performance.now() });
        object.position.copy(old.get(move.from)!);
      }
      this.pieces.add(object);
      if (move && game.isCheckmate() && p.type === 'k' && p.color === game.turn() && !this.reducedMotion) this.fallingKing = { object, start: performance.now(), shattered: false };
    }
  }

  private capture(victim: THREE.Object3D) {
    const position = victim.position.clone();
    victim.removeFromParent();
    victim.position.set(0, 0, 0);
    this.burst(position, victim, 24, ['#eacc8b', '#8ab7a0'], '#bd9342');
  }

  private burst(position: THREE.Vector3, victim: THREE.Object3D | null, count: number, colors: string[], emissive: string) {
    const group = new THREE.Group();
    group.position.copy(position);
    if (victim) group.add(victim);
    const shards: THREE.Mesh[] = [];
    for (let i = 0; i < count; i++) {
      const shard = mesh(new THREE.OctahedronGeometry(0.04 + (i % 4) * 0.02), new THREE.MeshStandardMaterial({ color: colors[i % colors.length], emissive, emissiveIntensity: 0.55, transparent: true, opacity: 1 }), 0, 0.4);
      shard.userData.angle = i * 2.399;
      shard.userData.speed = 0.6 + (i % 5) * 0.18;
      group.add(shard);
      shards.push(shard);
    }
    const ring = mesh(new THREE.RingGeometry(0.25, 0.29, 64), new THREE.MeshBasicMaterial({ color: '#f2cd76', transparent: true, opacity: 1, side: THREE.DoubleSide, depthWrite: false }), 0, 0.02);
    ring.rotation.x = -Math.PI / 2;
    group.add(ring);
    this.scene.add(group);
    this.captures.push({ group, victim, shards, ring, start: performance.now() });
  }

  private clearCaptures() {
    for (const effect of this.captures) this.disposeCapture(effect);
    this.captures = [];
  }

  private disposeCapture(effect: { group: THREE.Group; shards: THREE.Mesh[]; ring: THREE.Mesh }) {
    effect.group.traverse(object => { if (object instanceof THREE.Mesh) object.geometry.dispose(); });
    for (const object of [...effect.shards, effect.ring]) (object.material as THREE.Material).dispose();
    this.scene.remove(effect.group);
  }

  highlight(selected: Square | null, legal: Square[], last: string[] = []) {
    this.markers.traverse(object => {
      if (object instanceof THREE.Mesh) { object.geometry.dispose(); (object.material as THREE.Material).dispose(); }
    });
    this.markers.clear();
    for (const square of [...new Set([...last, ...(selected ? [selected] : []), ...legal])]) {
      const isLegal = legal.includes(square as Square);
      const geometry = isLegal ? new THREE.RingGeometry(0.1, 0.17, 32) : new THREE.PlaneGeometry(0.94, 0.94);
      const marker = mesh(geometry, new THREE.MeshBasicMaterial({ color: square === selected ? '#edc15e' : isLegal ? '#e6ba50' : '#e8ce80', transparent: true, opacity: isLegal ? 0.95 : 0.38, depthWrite: false }), square.charCodeAt(0) - 100.5, 0.216, 4.5 - Number(square[1]));
      marker.rotation.x = -Math.PI / 2;
      this.markers.add(marker);
    }
  }

  flip() { this.flipped = !this.flipped; this.resetCamera(); }
  toggleView() { this.overhead = !this.overhead; this.resetCamera(); return this.overhead; }
  resetCamera() {
    const sign = this.flipped ? -1 : 1;
    this.camera.position.set(this.overhead ? 0 : 10.3 * sign, this.overhead ? 23 : 13.4, this.overhead ? 0.01 * sign : 15.8 * sign);
    this.controls.target.set(0, 0.1, 0);
    this.controls.update();
  }
  private resize() {
    const { width, height } = this.container.getBoundingClientRect();
    this.camera.aspect = width / height;
    this.camera.zoom = Math.min(1, width / height / 1.05);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }
  private topple(king: { object: THREE.Object3D; start: number; shattered: boolean }, time: number) {
    const elapsed = time - king.start;
    const fall = Math.min(Math.max((elapsed - 600) / 300, 0), 1);
    const settle = Math.min(Math.max((elapsed - 900) / 400, 0), 1);
    king.object.rotation.x = Math.sin(elapsed * 0.09) * 0.06 * Math.min(elapsed / 600, 1) * (1 - fall);
    king.object.rotation.z = fall * fall * 1.45 - Math.sin(settle * Math.PI) * 0.12 * (1 - settle);
    if (fall === 1 && !king.shattered) {
      king.shattered = true;
      this.burst(king.object.position, null, 40, ['#7a1f2b', '#2b1a3a', '#e8c26a'], '#b0263a');
    }
    if (settle === 1) this.fallingKing = null;
  }

  private frame(time: number) {
    if (!this.reducedMotion) {
      this.flames.forEach((flame, i) => {
        flame.scale.y = flame.userData.height * (0.85 + Math.sin(time * 0.008 + i * 2.3) * 0.2);
        flame.rotation.z = Math.sin(time * 0.006 + i) * 0.15;
      });
      this.firelight.intensity = 24 + Math.sin(time * 0.012) * 3 + Math.sin(time * 0.019) * 2;
      this.dust.rotation.y = time * 0.000015;
    }
    this.animations = this.animations.filter(animation => {
      const t = Math.min((time - animation.start) / 550, 1);
      const eased = t * t * (3 - 2 * t);
      animation.object.position.lerpVectors(animation.from, animation.to, eased);
      animation.object.position.y += Math.sin(t * Math.PI) * 0.65;
      return t < 1;
    });
    if (this.fallingKing) this.topple(this.fallingKing, time);
    this.captures = this.captures.filter(effect => {
      const t = Math.min((time - effect.start) / 1150, 1);
      if (effect.victim) {
        effect.victim.position.y = t * 1.5;
        effect.victim.rotation.z = Math.sin(t * 24) * t * 0.3;
        effect.victim.rotation.y = t * 5;
        effect.victim.scale.setScalar(Math.max(0, 1 - Math.max(0, t - 0.18) * 2.8));
      }
      effect.shards.forEach((shard, i) => {
        const distance = t * shard.userData.speed * 2;
        shard.position.set(Math.cos(shard.userData.angle) * distance, 0.4 + Math.sin(t * Math.PI) * (1 + i % 3 * 0.3), Math.sin(shard.userData.angle) * distance);
        shard.rotation.set(t * (i + 1), t * 5, t * 3);
        (shard.material as THREE.MeshStandardMaterial).opacity = 1 - t;
      });
      effect.ring.scale.setScalar(1 + t * 7);
      (effect.ring.material as THREE.MeshBasicMaterial).opacity = 1 - t;
      if (t >= 1) this.disposeCapture(effect);
      return t < 1;
    });
    this.controls.update();
    this.library.visible = this.camera.position.z > -7;
    this.sideLibrary.visible = this.camera.position.x > -9;
    this.renderer.render(this.scene, this.camera);
  }
}
