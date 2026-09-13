import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Chess, type Square, type PieceSymbol, type Move } from 'chess.js';
import { gold, jade, stone, mesh } from './parts';
import { buildLibrary, buildGreatHall, buildOffice, type Background } from './rooms';
import { applyPieceStyle, type PieceStyle } from './pieceStyles';

export const pieceColors = { white: '#f9ecd2', green: '#31574d', brown: '#4b2c1a', black: '#20272a', blue: '#1f3a6e', orange: '#c7772e', salmon: '#e8907a', gray: '#8a8d8f' } as const;
export type PieceColor = keyof typeof pieceColors;

function cylinder(group: THREE.Group, top: number, bottom: number, height: number, y: number, material: THREE.Material) {
  group.add(mesh(new THREE.CylinderGeometry(top, bottom, height, 32), material, 0, y));
}

function piece(type: PieceSymbol, material: THREE.Material): THREE.Group {
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
  private playerMaterial = new THREE.MeshPhysicalMaterial();
  private cpuMaterial = new THREE.MeshPhysicalMaterial();
  private rooms = { library: buildLibrary(), greatHall: buildGreatHall(), office: buildOffice() };
  private room: Background = 'library';
  private markers = new THREE.Group();
  private tiles: THREE.Mesh[] = [];
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
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, matchMedia('(pointer: coarse)').matches ? 1.5 : 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.setClearColor('#eee9df', 0);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;
    this.renderer.domElement.setAttribute('aria-label', 'Interactive 3D chessboard. Select a piece, then a highlighted square. You can also use the move input below.');
    this.renderer.domElement.setAttribute('role', 'img');
    container.prepend(this.renderer.domElement);
    const environment = new THREE.PMREMGenerator(this.renderer).fromScene(new RoomEnvironment(), 0.04).texture;
    this.playerMaterial.envMap = this.cpuMaterial.envMap = environment;
    this.setPieceStyle('classic');
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
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > (e.pointerType === 'touch' ? 14 : 6)) return;
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
    for (const room of Object.values(this.rooms)) this.scene.add(room.back, room.side);
  }

  setBackground(background: Background) {
    this.room = background;
  }

  setPieceStyle(style: PieceStyle) {
    applyPieceStyle(this.playerMaterial, style);
    applyPieceStyle(this.cpuMaterial, style);
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
    this.camera.zoom = Math.min(Math.min(1.45, Math.max(1, 560 / height)), width / height / 1.05);
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
      this.rooms[this.room].animate(time);
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
    for (const [name, room] of Object.entries(this.rooms)) {
      room.back.visible = name === this.room && this.camera.position.z > -7;
      room.side.visible = name === this.room && this.camera.position.x > -9;
    }
    this.renderer.render(this.scene, this.camera);
  }
}
