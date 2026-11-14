import {
  BoxGeometry,
  CapsuleGeometry,
  type CanvasTexture,
  CatmullRomCurve3,
  ConeGeometry,
  CylinderGeometry,
  DodecahedronGeometry,
  ExtrudeGeometry,
  Group,
  Mesh,
  MeshStandardMaterial,
  OctahedronGeometry,
  Path,
  PointLight,
  RepeatWrapping,
  Shape,
  SphereGeometry,
  TorusGeometry,
  type Texture,
  TubeGeometry,
  Vector3,
} from "three";
import { linear, paint, seeded, toTexture } from "../art/canvas";

export interface HouseParts {
  group: Group;
  walls: Group;
  roof: Group;
}

function repeat(texture: CanvasTexture | Texture, x: number, y: number): Texture {
  const t = texture.clone();
  t.wrapS = RepeatWrapping;
  t.wrapT = RepeatWrapping;
  t.repeat.set(x, y);
  t.needsUpdate = true;
  return t;
}

function strokes(base: string, colors: string[], vertical: boolean, thick: number, seed: number): CanvasTexture {
  const rand = seeded(seed);
  return toTexture(
    paint(256, 256, (ctx, w, h) => {
      ctx.fillStyle = base;
      ctx.fillRect(0, 0, w, h);
      for (let i = 0; i < 260; i++) {
        ctx.strokeStyle = colors[Math.floor(rand() * colors.length)] ?? base;
        ctx.lineWidth = thick * (0.5 + rand());
        const a = rand() * w;
        const b = rand() * h;
        ctx.beginPath();
        if (vertical) {
          ctx.moveTo(a, b - 60);
          ctx.lineTo(a + (rand() - 0.5) * 10, b + 60);
        } else {
          ctx.moveTo(a - 60, b);
          ctx.lineTo(a + 60, b + (rand() - 0.5) * 10);
        }
        ctx.stroke();
      }
    }),
  );
}

export const strawTexture = (): CanvasTexture => strokes("#e8bf55", ["#f7dc8a", "#c9962e", "#fff0b3", "#b07f22"], true, 3, 11);

function logTexture(): CanvasTexture {
  return toTexture(
    paint(256, 256, (ctx, w, h) => {
      for (let x = 0; x < w; x += 32) {
        const g = ctx.createLinearGradient(x, 0, x + 32, 0);
        g.addColorStop(0, "#5a3519");
        g.addColorStop(0.5, "#a06a3a");
        g.addColorStop(1, "#4a2a12");
        ctx.fillStyle = g;
        ctx.fillRect(x, 0, 32, h);
        ctx.fillStyle = "rgba(40,20,5,0.5)";
        for (let y = 20; y < h; y += 70) ctx.fillRect(x + 10, y + (x % 50), 10, 6);
      }
    }),
  );
}

function plasterTexture(): CanvasTexture {
  return strokes("#fbf3e4", ["#efe2c9", "#ffffff"], false, 6, 5);
}

const ink = new MeshStandardMaterial({ color: "#3b2416", roughness: 0.8 });
const glow = new MeshStandardMaterial({ color: "#ffdc7a", emissive: "#ffb33a", emissiveIntensity: 1.6 });

function door(width: number, height: number, z: number, color = "#7a3b1e"): Group {
  const g = new Group();
  const panel = new Mesh(new BoxGeometry(width, height, 0.05), new MeshStandardMaterial({ color, roughness: 0.6 }));
  panel.position.set(0, height / 2, z);
  const knob = new Mesh(new SphereGeometry(0.035, 12, 12), new MeshStandardMaterial({ color: "#ffd23f", metalness: 0.6, roughness: 0.3 }));
  knob.position.set(width * 0.3, height * 0.48, z + 0.04);
  g.add(panel, knob);
  return g;
}

function windowPane(size: number, x: number, y: number, z: number): Group {
  const g = new Group();
  const frame = new Mesh(new BoxGeometry(size + 0.08, size + 0.08, 0.04), ink);
  const pane = new Mesh(new BoxGeometry(size, size, 0.05), glow);
  const barV = new Mesh(new BoxGeometry(0.03, size, 0.07), ink);
  const barH = new Mesh(new BoxGeometry(size, 0.03, 0.07), ink);
  g.add(frame, pane, barV, barH);
  g.position.set(x, y, z);
  return g;
}

export function cottage(): HouseParts {
  const walls = new Group();
  const body = new Mesh(new BoxGeometry(1.6, 1.1, 1.2), new MeshStandardMaterial({ map: plasterTexture(), roughness: 0.95 }));
  body.position.y = 0.55;
  const beams = new MeshStandardMaterial({ color: "#5a3519" });
  for (const x of [-0.79, 0.79]) {
    const beam = new Mesh(new BoxGeometry(0.06, 1.1, 1.22), beams);
    beam.position.set(x, 0.55, 0);
    walls.add(beam);
  }
  walls.add(body, door(0.38, 0.7, 0.61, "#2f6b8f"), windowPane(0.28, -0.48, 0.66, 0.61), windowPane(0.28, 0.48, 0.66, 0.61));
  const roof = new Group();
  const shape = new Shape();
  shape.moveTo(-1.0, 0);
  shape.lineTo(1.0, 0);
  shape.lineTo(0, 0.9);
  shape.closePath();
  const thatch = new Mesh(new ExtrudeGeometry(shape, { depth: 1.4, bevelEnabled: true, bevelSize: 0.06, bevelThickness: 0.06, bevelSegments: 2 }), new MeshStandardMaterial({ map: repeat(strawTexture(), 1, 1), roughness: 1 }));
  thatch.position.set(0, 1.1, -0.7);
  const chimney = new Mesh(new BoxGeometry(0.26, 0.6, 0.26), new MeshStandardMaterial({ color: "#a65a3a" }));
  chimney.position.set(-0.5, 1.75, 0);
  roof.add(thatch, chimney);
  const group = new Group();
  group.add(walls, roof);
  return { group, walls, roof };
}

export function rock(scale: number, seed: number): Mesh {
  const rand = seeded(seed);
  const mesh = new Mesh(new DodecahedronGeometry(scale, 0), new MeshStandardMaterial({ color: "#9a958c", roughness: 1, flatShading: true }));
  mesh.scale.set(1 + rand() * 0.4, 0.6 + rand() * 0.3, 1 + rand() * 0.3);
  mesh.position.y = scale * 0.4;
  return mesh;
}

export function mushroom(): Group {
  const g = new Group();
  const stem = new Mesh(new CylinderGeometry(0.05, 0.07, 0.2, 10), new MeshStandardMaterial({ color: "#fff4dc" }));
  stem.position.y = 0.1;
  const cap = new Mesh(new SphereGeometry(0.13, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2), new MeshStandardMaterial({ color: "#e0413a", roughness: 0.5 }));
  cap.position.y = 0.18;
  g.add(stem, cap);
  for (let i = 0; i < 4; i++) {
    const dot = new Mesh(new SphereGeometry(0.022, 8, 8), new MeshStandardMaterial({ color: "#ffffff" }));
    const a = i * 1.6;
    dot.position.set(Math.cos(a) * 0.08, 0.27, Math.sin(a) * 0.08);
    g.add(dot);
  }
  return g;
}

export function logPile(): Group {
  const g = new Group();
  const bark = new MeshStandardMaterial({ map: repeat(logTexture(), 1, 1), roughness: 0.9 });
  const rows = [[-0.2, 0.09], [0, 0.09], [0.2, 0.09], [-0.1, 0.25], [0.1, 0.25], [0, 0.41]] as const;
  for (const [x, y] of rows) {
    const log = new Mesh(new CylinderGeometry(0.09, 0.09, 0.6, 12), bark);
    log.rotation.x = Math.PI / 2;
    log.position.set(x, y, 0);
    g.add(log);
  }
  return g;
}

export function fence(length: number): Group {
  const g = new Group();
  const wood = new MeshStandardMaterial({ color: "#d9b07a", roughness: 0.8 });
  const posts = Math.round(length / 0.35);
  for (let i = 0; i <= posts; i++) {
    const post = new Mesh(new BoxGeometry(0.07, 0.42, 0.05), wood);
    post.position.set(-length / 2 + (i * length) / posts, 0.21, 0);
    g.add(post);
  }
  for (const y of [0.14, 0.32]) {
    const rail = new Mesh(new BoxGeometry(length, 0.05, 0.03), wood);
    rail.position.set(0, y, 0.03);
    g.add(rail);
  }
  return g;
}


export function logsFire(): Group {
  const g = new Group();
  const bark = new MeshStandardMaterial({ color: "#4a2a12", roughness: 0.9 });
  for (let i = 0; i < 4; i++) {
    const log = new Mesh(new CylinderGeometry(0.07, 0.07, 0.7, 10), bark);
    log.rotation.z = Math.PI / 2;
    log.rotation.y = (i / 4) * Math.PI;
    log.position.y = 0.07;
    g.add(log);
  }
  const embers = new Mesh(new SphereGeometry(0.3, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2), new MeshStandardMaterial({ color: "#ff6a1f", emissive: "#ff4a10", emissiveIntensity: 2 }));
  embers.scale.y = 0.4;
  g.add(embers);
  return g;
}


export interface Oven {
  group: Group;
  door: Group;
}

export function oven(bricks: Texture): Oven {
  const g = new Group();
  const brick = new MeshStandardMaterial({ map: repeat(bricks, 1.6, 1.4), roughness: 0.85, color: "#d8b8a0" });
  const outline = new Shape();
  outline.moveTo(-1.1, 0);
  outline.lineTo(1.1, 0);
  outline.lineTo(1.1, 1.7);
  outline.quadraticCurveTo(0, 2.4, -1.1, 1.7);
  outline.closePath();
  const mouth = new Path();
  mouth.moveTo(-0.5, 0.3);
  mouth.lineTo(0.5, 0.3);
  mouth.lineTo(0.5, 0.85);
  mouth.absarc(0, 0.85, 0.5, 0, Math.PI, false);
  mouth.closePath();
  outline.holes.push(mouth);
  const front = new Mesh(new ExtrudeGeometry(outline, { depth: 0.25, bevelEnabled: false }), brick);
  front.position.z = 0.35;
  const body = new Mesh(new BoxGeometry(2.2, 1.9, 0.8), brick);
  body.position.set(0, 0.95, -0.05);
  const hollow = new Mesh(new BoxGeometry(0.96, 1.05, 0.5), new MeshStandardMaterial({ color: "#1c1210", emissive: "#ff4a10", emissiveIntensity: 0.25 }));
  hollow.position.set(0, 0.84, 0.28);
  const fire = logsFire();
  fire.scale.setScalar(0.75);
  fire.position.set(0, 0.32, 0.3);
  const chimney = new Mesh(new BoxGeometry(0.5, 1.1, 0.5), brick);
  chimney.position.set(0.55, 2.2, -0.1);
  const door = new Group();
  const plate = new Mesh(new BoxGeometry(1.04, 1.12, 0.06), new MeshStandardMaterial({ color: "#2a2a30", metalness: 0.75, roughness: 0.35 }));
  plate.position.set(0.52, 0.86, 0);
  const handle = new Mesh(new BoxGeometry(0.06, 0.3, 0.08), new MeshStandardMaterial({ color: "#c9962e", metalness: 0.8, roughness: 0.3 }));
  handle.position.set(0.9, 0.86, 0.06);
  door.add(plate, handle);
  door.position.set(-0.52, 0, 0.64);
  g.add(body, front, hollow, fire, chimney, door);
  return { group: g, door };
}

export interface Chest {
  group: Group;
  lid: Group;
  glow: PointLight;
}

function plankTexture(): CanvasTexture {
  const rand = seeded(17);
  return toTexture(
    paint(256, 256, (ctx, w, h) => {
      const plank = h / 4;
      for (let i = 0; i < 4; i++) {
        const y = i * plank;
        ctx.fillStyle = linear(ctx, y, y + plank, ["#b06a32", "#8a4a20", "#6e3818"]);
        ctx.fillRect(0, y, w, plank);
        ctx.strokeStyle = "rgba(60,25,8,0.35)";
        ctx.lineWidth = 2;
        for (let k = 0; k < 5; k++) {
          const gy = y + 8 + rand() * (plank - 16);
          ctx.beginPath();
          ctx.moveTo(0, gy);
          ctx.bezierCurveTo(w * 0.3, gy + (rand() - 0.5) * 8, w * 0.7, gy + (rand() - 0.5) * 8, w, gy);
          ctx.stroke();
        }
        ctx.fillStyle = "#3b1d0c";
        ctx.fillRect(0, y + plank - 4, w, 4);
      }
    }),
  );
}

export function treasureChest(): Chest {
  const W = 0.9;
  const H = 0.42;
  const D = 0.56;
  const R = D / 2;
  const T = 0.04;
  const g = new Group();
  const wood = new MeshStandardMaterial({ map: plankTexture(), roughness: 0.75, side: 2 });
  const gold = new MeshStandardMaterial({ color: "#f5b82e", emissive: "#8a5a00", emissiveIntensity: 0.35, metalness: 0.85, roughness: 0.28 });
  const dark = new MeshStandardMaterial({ color: "#1d120a", roughness: 0.9 });
  const walls: [number, number, number, number, number, number][] = [
    [W, H, T, 0, H / 2, D / 2 - T / 2],
    [W, H, T, 0, H / 2, -D / 2 + T / 2],
    [T, H, D, W / 2 - T / 2, H / 2, 0],
    [T, H, D, -W / 2 + T / 2, H / 2, 0],
    [W, T, D, 0, T / 2, 0],
  ];
  for (const [w, h, d, x, y, z] of walls) {
    const wall = new Mesh(new BoxGeometry(w, h, d), wood);
    wall.position.set(x, y, z);
    g.add(wall);
  }
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    const post = new Mesh(new BoxGeometry(0.06, H + 0.01, 0.06), gold);
    post.position.set(sx * W / 2, H / 2, sz * D / 2);
    g.add(post);
  }
  for (const y of [0.025, H - 0.02]) {
    for (const sz of [-1, 1]) {
      const rim = new Mesh(new BoxGeometry(W + 0.02, 0.04, 0.05), gold);
      rim.position.set(0, y, sz * D / 2);
      g.add(rim);
    }
    for (const sx of [-1, 1]) {
      const rim = new Mesh(new BoxGeometry(0.05, 0.04, D + 0.02), gold);
      rim.position.set(sx * W / 2, y, 0);
      g.add(rim);
    }
  }
  for (const x of [-0.27, 0.27]) {
    const strap = new Mesh(new BoxGeometry(0.05, H, 0.02), gold);
    strap.position.set(x, H / 2, D / 2 + 0.012);
    g.add(strap);
  }
  const plate = new Mesh(new BoxGeometry(0.13, 0.15, 0.03), gold);
  plate.position.set(0, H - 0.1, D / 2 + 0.018);
  const hole = new Mesh(new CylinderGeometry(0.016, 0.016, 0.02, 12), dark);
  hole.rotation.x = Math.PI / 2;
  hole.position.set(0, H - 0.085, D / 2 + 0.035);
  const slot = new Mesh(new BoxGeometry(0.012, 0.04, 0.02), dark);
  slot.position.set(0, H - 0.115, D / 2 + 0.035);
  g.add(plate, hole, slot);
  const bed = new Mesh(new BoxGeometry(W - 2 * T, 0.02, D - 2 * T), gold);
  bed.position.y = H - 0.07;
  g.add(bed);
  const rand = seeded(13);
  const coin = new CylinderGeometry(0.042, 0.042, 0.012, 18);
  for (let i = 0; i < 110; i++) {
    const x = (rand() - 0.5) * (W - 0.14);
    const z = (rand() - 0.5) * (D - 0.14);
    const mound = Math.max(0, 1 - (x / (W / 2)) ** 2 - (z / (D / 2)) ** 2);
    const piece = new Mesh(coin, gold);
    piece.position.set(x, H - 0.055 + mound * 0.13 * (0.6 + rand() * 0.4), z);
    piece.rotation.set((rand() - 0.5) * 1.1, rand() * Math.PI, (rand() - 0.5) * 1.1);
    g.add(piece);
  }
  const gems = ["#e8203c", "#2f9fe8", "#3ccf6a", "#b04ae8", "#ffffff"];
  for (let i = 0; i < 7; i++) {
    const color = gems[i % gems.length] ?? "#e8203c";
    const gem = new Mesh(new OctahedronGeometry(0.045, 0), new MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35, metalness: 0.2, roughness: 0.1, flatShading: true }));
    const x = (rand() - 0.5) * (W - 0.3);
    const z = (rand() - 0.5) * (D - 0.25);
    gem.position.set(x, H + 0.06 * Math.max(0, 1 - (x / (W / 2)) ** 2), z);
    gem.rotation.set(rand(), rand(), rand());
    g.add(gem);
  }
  const necklace = new CatmullRomCurve3([
    new Vector3(-0.22, H + 0.03, 0.02),
    new Vector3(-0.1, H + 0.03, D / 2 - 0.02),
    new Vector3(0.02, H - 0.03, D / 2 + 0.04),
    new Vector3(0.14, H - 0.16, D / 2 + 0.045),
    new Vector3(0.24, H - 0.05, D / 2 + 0.04),
    new Vector3(0.3, H + 0.02, D / 2 - 0.04),
  ]);
  const pearl = new MeshStandardMaterial({ color: "#fff8ee", roughness: 0.15, metalness: 0.1 });
  for (const point of necklace.getSpacedPoints(26)) {
    const bead = new Mesh(new SphereGeometry(0.016, 10, 8), pearl);
    bead.position.copy(point);
    g.add(bead);
  }
  for (let i = 0; i < 7; i++) {
    const loose = new Mesh(coin, gold);
    loose.position.set(-0.35 + rand() * 0.7, 0.007 + (i === 2 ? 0.013 : 0), D / 2 + 0.1 + rand() * 0.22);
    loose.rotation.y = rand() * Math.PI;
    g.add(loose);
  }
  const lid = new Group();
  const shell = new Mesh(new CylinderGeometry(R, R, W, 28, 1, false, 0, Math.PI), wood);
  shell.rotation.z = Math.PI / 2;
  shell.position.set(0, 0, R);
  lid.add(shell);
  for (const x of [-W / 2 + 0.02, -0.27, 0.27, W / 2 - 0.02]) {
    const band = new Mesh(new CylinderGeometry(R + 0.012, R + 0.012, 0.05, 28, 1, true, 0, Math.PI), gold);
    band.material.side = 2;
    band.rotation.z = Math.PI / 2;
    band.position.set(x, 0, R);
    lid.add(band);
  }
  const hasp = new Mesh(new BoxGeometry(0.08, 0.1, 0.025), gold);
  hasp.position.set(0, -0.02, D + 0.018);
  lid.add(hasp);
  lid.position.set(0, H, -D / 2);
  g.add(lid);
  const glow = new PointLight("#ffc85a", 0, 2.2, 1.5);
  glow.position.set(0, H + 0.35, 0.1);
  g.add(glow);
  return { group: g, lid, glow };
}

export function stump(): Group {
  const g = new Group();
  const trunk = new Mesh(new CylinderGeometry(0.3, 0.36, 0.4, 16), new MeshStandardMaterial({ map: repeat(logTexture(), 2, 1), roughness: 0.9 }));
  trunk.position.y = 0.2;
  const rings = new Mesh(new CylinderGeometry(0.29, 0.29, 0.02, 16), new MeshStandardMaterial({ color: "#e8c48a", roughness: 0.9 }));
  rings.position.y = 0.41;
  const handle = new Mesh(new CylinderGeometry(0.025, 0.03, 0.6, 8), new MeshStandardMaterial({ color: "#a8743a" }));
  handle.position.set(0.12, 0.66, 0);
  handle.rotation.z = -0.5;
  const head = new Mesh(new BoxGeometry(0.2, 0.12, 0.03), new MeshStandardMaterial({ color: "#c8ccd4", metalness: 0.8, roughness: 0.3 }));
  head.position.set(0.0, 0.45, 0);
  head.rotation.z = -0.5;
  g.add(trunk, rings, handle, head);
  return g;
}


export interface Beanstalk {
  group: Group;
  grow: (amount: number, leaves: (spot: number) => number) => void;
  tip: () => Vector3;
}

export function stalkPoint(t: number, height: number, girth: number): Vector3 {
  const turns = 9 * Math.max(1, height / 6);
  const swirl = 0.35 * girth * (1 - t * 0.5);
  return new Vector3(Math.sin(t * turns) * swirl, t * height, Math.cos(t * turns) * swirl * 0.6);
}

export function beanstalk(height: number, spots: number[], girth = 1): Beanstalk {
  const group = new Group();
  const leafSize = girth > 1 ? girth * 0.5 : 1;
  const points = Array.from({ length: 40 }, (_, i) => stalkPoint(i / 39, height, girth));
  const curve = new CatmullRomCurve3(points);
  const green = new MeshStandardMaterial({ color: "#3fa83a", roughness: 0.6 });
  const tubular = 400;
  const radial = girth > 1 ? 18 : 10;
  const stalk = new Mesh(new TubeGeometry(curve, tubular, 0.09 * girth, radial, false), green);
  const vine = new Mesh(new TubeGeometry(new CatmullRomCurve3(points.map((p, i) => p.clone().add(new Vector3(Math.cos(i * 2) * 0.1 * girth, 0, Math.sin(i * 2) * 0.1 * girth)))), tubular, 0.04 * girth, 8, false), new MeshStandardMaterial({ color: "#7ad84a", roughness: 0.6 }));
  group.add(stalk, vine);
  const leafMaterial = new MeshStandardMaterial({ color: "#5ccf4a", roughness: 0.5, side: 2 });
  const leaves = spots.map((spot, i) => {
    const pivot = new Group();
    pivot.position.copy(curve.getPoint(spot));
    pivot.rotation.y = i * 2.3;
    const leaf = new Mesh(new SphereGeometry(0.3 * leafSize, 16, 10), leafMaterial);
    leaf.scale.set(1, 0.12, 0.5);
    leaf.position.x = 0.3 * leafSize + 0.09 * girth;
    leaf.rotation.z = 0.35;
    pivot.add(leaf);
    group.add(pivot);
    return pivot;
  });
  const pod = new Mesh(new ConeGeometry(0.09 * girth, 0.5 * girth, 16), new MeshStandardMaterial({ color: "#8fe36a", emissive: "#6fbf3a", emissiveIntensity: 0.5 }));
  group.add(pod);
  const full = (geometry: TubeGeometry) => geometry.index?.count ?? 0;
  const grow = (amount: number, open: (spot: number) => number) => {
    for (const mesh of [stalk, vine]) mesh.geometry.setDrawRange(0, Math.floor((full(mesh.geometry) * amount) / 6) * 6);
    leaves.forEach((pivot, i) => {
      const s = Math.max(open(spots[i] ?? 0), 0.0001);
      pivot.scale.setScalar(s);
      pivot.visible = s > 0.01;
    });
    pod.position.copy(curve.getPoint(Math.max(amount, 0.001)));
    pod.visible = amount > 0;
  };
  return { group, grow, tip: () => pod.position.clone() };
}

export function cloudPuff(seed: number, scale = 1): Group {
  const rand = seeded(seed);
  const g = new Group();
  const white = new MeshStandardMaterial({ color: "#ffffff", roughness: 0.95, });
  for (let i = 0; i < 6; i++) {
    const r = (0.22 + rand() * 0.16) * scale;
    const puff = new Mesh(new SphereGeometry(r, 16, 12), white);
    puff.position.set((i - 2.5) * 0.24 * scale, r * 0.6 + rand() * 0.08 * scale, (rand() - 0.5) * 0.25 * scale);
    g.add(puff);
  }
  return g;
}

export interface Castle {
  group: Group;
  door: Group;
}

export function castle(bricks: Texture): Castle {
  const g = new Group();
  const stone = new MeshStandardMaterial({ map: repeat(bricks, 2, 1.5), color: "#c8c4d8", roughness: 0.9 });
  const roofs = new MeshStandardMaterial({ color: "#6a3aa8", roughness: 0.6 });
  const keep = new Mesh(new BoxGeometry(1.8, 1.5, 1.0), stone);
  keep.position.y = 0.75;
  g.add(keep);
  for (let i = 0; i < 6; i++) {
    const merlon = new Mesh(new BoxGeometry(0.18, 0.2, 1.0), stone);
    merlon.position.set(-0.75 + i * 0.3, 1.6, 0);
    g.add(merlon);
  }
  for (const [x, h] of [[-1.05, 2.2], [1.05, 2.2], [0, 2.7]] as const) {
    const tower = new Mesh(new CylinderGeometry(0.32, 0.36, h, 20), stone);
    tower.position.set(x, h / 2, x === 0 ? -0.35 : 0.1);
    const roof = new Mesh(new ConeGeometry(0.44, 0.8, 20), roofs);
    roof.position.set(x, h + 0.4, tower.position.z);
    const flag = new Mesh(new BoxGeometry(0.22, 0.13, 0.01), new MeshStandardMaterial({ color: "#ffd23f", side: 2 }));
    flag.position.set(x + 0.12, h + 0.9, tower.position.z);
    const pole = new Mesh(new CylinderGeometry(0.01, 0.01, 0.3, 6), ink);
    pole.position.set(x, h + 0.85, tower.position.z);
    g.add(tower, roof, flag, pole);
    if (x !== 0) g.add(windowPane(0.18, x, h * 0.7, 0.45));
  }
  const arch = new Mesh(new CylinderGeometry(0.34, 0.34, 0.06, 24, 1, false, 0, Math.PI), new MeshStandardMaterial({ color: "#1d140c" }));
  arch.rotation.x = Math.PI / 2;
  arch.rotation.z = Math.PI / 2;
  arch.position.set(0, 0.7, 0.5);
  const hole = new Mesh(new BoxGeometry(0.68, 0.7, 0.06), new MeshStandardMaterial({ color: "#1d140c" }));
  hole.position.set(0, 0.35, 0.5);
  const door = new Group();
  const plank = new Mesh(new BoxGeometry(0.66, 0.98, 0.05), new MeshStandardMaterial({ color: "#7a3b1e", roughness: 0.7 }));
  plank.position.set(0.33, 0.49, 0);
  const knocker = new Mesh(new TorusGeometry(0.05, 0.012, 8, 16), new MeshStandardMaterial({ color: "#ffd23f", metalness: 0.7, roughness: 0.3 }));
  knocker.position.set(0.52, 0.5, 0.04);
  door.add(plank, knocker);
  door.position.set(-0.33, 0, 0.54);
  g.add(arch, hole, door, windowPane(0.22, -0.55, 1.05, 0.52), windowPane(0.22, 0.55, 1.05, 0.52));
  return { group: g, door };
}

const goldMaterial = new MeshStandardMaterial({ color: "#f5b82e", emissive: "#8a5a00", emissiveIntensity: 0.45, metalness: 0.85, roughness: 0.22 });

export function goldenEgg(): Mesh {
  const egg = new Mesh(new SphereGeometry(0.09, 20, 14), goldMaterial);
  egg.scale.set(1, 1.3, 1);
  egg.castShadow = true;
  return egg;
}

export function giantTable(): Group {
  const g = new Group();
  const wood = new MeshStandardMaterial({ map: repeat(logTexture(), 2, 1), roughness: 0.8 });
  const top = new Mesh(new BoxGeometry(2.4, 0.12, 1.1), wood);
  top.position.y = 0.9;
  g.add(top);
  for (const [x, z] of [[-1.05, -0.45], [1.05, -0.45], [-1.05, 0.45], [1.05, 0.45]] as const) {
    const leg = new Mesh(new CylinderGeometry(0.08, 0.1, 0.9, 10), wood);
    leg.position.set(x, 0.45, z);
    g.add(leg);
  }
  const nest = new Mesh(new TorusGeometry(0.22, 0.08, 10, 20), new MeshStandardMaterial({ color: "#c9962e", roughness: 1 }));
  nest.rotation.x = Math.PI / 2;
  nest.position.set(-0.5, 1.0, 0);
  g.add(nest);
  return g;
}

export function bean(): Mesh {
  const mesh = new Mesh(new SphereGeometry(0.05, 12, 10), new MeshStandardMaterial({ color: "#b6ff7a", emissive: "#7ad84a", emissiveIntensity: 1.4 }));
  mesh.scale.set(1, 1.4, 1);
  return mesh;
}

export interface Giant {
  group: Group;
  body: Group;
  head: Group;
  jaw: Mesh;
  arms: [Group, Group];
  legs: [Group, Group];
  lids: Mesh[];
}

function limb3d(radius: number, length: number, material: MeshStandardMaterial, end: Mesh): Group {
  const pivot = new Group();
  const bone = new Mesh(new CapsuleGeometry(radius, length, 8, 16), material);
  bone.position.y = -length / 2 - radius * 0.5;
  end.position.y = -length - radius;
  pivot.add(bone, end);
  return pivot;
}

export function giant3d(): Giant {
  const skin = new MeshStandardMaterial({ color: "#f2b48c", roughness: 0.6 });
  const tunic = new MeshStandardMaterial({ color: "#c8502a", roughness: 0.7 });
  const pants = new MeshStandardMaterial({ color: "#4a3a6a", roughness: 0.8 });
  const hair = new MeshStandardMaterial({ color: "#2a1d12", roughness: 0.9 });
  const boot = new MeshStandardMaterial({ color: "#3b2416", roughness: 0.7 });
  const white = new MeshStandardMaterial({ color: "#ffffff", roughness: 0.3 });
  const group = new Group();
  const body = new Group();
  body.position.y = 1.35;
  const torso = new Mesh(new CapsuleGeometry(0.55, 0.6, 10, 24), tunic);
  torso.position.y = 0.55;
  torso.scale.set(1, 1, 0.8);
  const belt = new Mesh(new TorusGeometry(0.56, 0.07, 10, 32), new MeshStandardMaterial({ color: "#3b2416" }));
  belt.rotation.x = Math.PI / 2;
  belt.scale.set(1, 0.8, 1);
  belt.position.y = 0.2;
  const buckle = new Mesh(new BoxGeometry(0.2, 0.16, 0.06), goldMaterial);
  buckle.position.set(0, 0.2, 0.45);
  const head = new Group();
  head.position.y = 1.45;
  const skull = new Mesh(new SphereGeometry(0.42, 32, 24), skin);
  const beard = new Mesh(new SphereGeometry(0.4, 24, 16), hair);
  beard.scale.set(1, 1.05, 0.7);
  beard.position.set(0, -0.28, 0.16);
  const cap = new Mesh(new SphereGeometry(0.44, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2.2), hair);
  cap.position.y = 0.04;
  const nose = new Mesh(new SphereGeometry(0.1, 16, 12), new MeshStandardMaterial({ color: "#e07a5a", roughness: 0.5 }));
  nose.position.set(0, -0.02, 0.42);
  const jaw = new Mesh(new SphereGeometry(0.13, 16, 12), new MeshStandardMaterial({ color: "#5e1f2c" }));
  jaw.scale.set(1.4, 0.3, 0.5);
  jaw.position.set(0, -0.2, 0.4);
  const lids: Mesh[] = [];
  head.add(skull, beard, cap, nose, jaw);
  for (const side of [-1, 1]) {
    const eye = new Mesh(new SphereGeometry(0.08, 16, 12), white);
    eye.position.set(side * 0.15, 0.1, 0.36);
    const pupil = new Mesh(new SphereGeometry(0.04, 12, 8), new MeshStandardMaterial({ color: "#1d140c" }));
    pupil.position.set(side * 0.15, 0.1, 0.43);
    const lid = new Mesh(new SphereGeometry(0.085, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2), skin);
    lid.position.copy(eye.position);
    lid.rotation.x = 0.4;
    lid.visible = false;
    const brow = new Mesh(new BoxGeometry(0.18, 0.05, 0.05), hair);
    brow.position.set(side * 0.15, 0.24, 0.38);
    brow.rotation.z = side * -0.35;
    const ear = new Mesh(new SphereGeometry(0.1, 12, 8), skin);
    ear.position.set(side * 0.42, 0, 0);
    ear.scale.set(0.5, 1, 0.8);
    head.add(eye, pupil, lid, brow, ear);
    lids.push(lid);
  }
  const arms = [-1, 1].map((side) => {
    const arm = limb3d(0.17, 0.75, tunic, new Mesh(new SphereGeometry(0.2, 16, 12), skin));
    arm.position.set(side * 0.72, 1.0, 0);
    return arm;
  }) as [Group, Group];
  body.add(torso, belt, buckle, head, ...arms);
  const legs = [-1, 1].map((side) => {
    const foot = new Mesh(new BoxGeometry(0.34, 0.2, 0.55), boot);
    foot.position.z = 0.1;
    const leg = limb3d(0.22, 1.0, pants, foot);
    leg.position.set(side * 0.3, 1.35, 0);
    return leg;
  }) as [Group, Group];
  group.add(body, ...legs);
  return { group, body, head, jaw, arms, legs, lids };
}

export interface Hen {
  group: Group;
  body: Group;
  head: Group;
}

export function hen3d(): Hen {
  const feathers = new MeshStandardMaterial({ color: "#ffe9a0", emissive: "#a87a10", emissiveIntensity: 0.25, roughness: 0.5, metalness: 0.2 });
  const red = new MeshStandardMaterial({ color: "#e8413c" });
  const group = new Group();
  const body = new Group();
  const torso = new Mesh(new SphereGeometry(0.22, 24, 16), feathers);
  torso.scale.set(1.2, 0.95, 0.9);
  const tail = new Mesh(new ConeGeometry(0.12, 0.3, 12), feathers);
  tail.position.set(-0.26, 0.12, 0);
  tail.rotation.z = 0.9;
  body.add(torso, tail);
  for (const side of [-1, 1]) {
    const wingMesh = new Mesh(new SphereGeometry(0.13, 16, 10), feathers);
    wingMesh.scale.set(1.3, 0.7, 0.4);
    wingMesh.position.set(-0.02, 0.02, side * 0.19);
    body.add(wingMesh);
  }
  const head = new Group();
  head.position.set(0.24, 0.22, 0);
  const skull = new Mesh(new SphereGeometry(0.11, 16, 12), feathers);
  const beak = new Mesh(new ConeGeometry(0.04, 0.1, 10), new MeshStandardMaterial({ color: "#ff9a3d" }));
  beak.rotation.z = -Math.PI / 2;
  beak.position.set(0.13, -0.01, 0);
  const wattle = new Mesh(new SphereGeometry(0.03, 8, 8), red);
  wattle.position.set(0.09, -0.08, 0);
  head.add(skull, beak, wattle);
  for (let i = 0; i < 3; i++) {
    const comb = new Mesh(new SphereGeometry(0.04, 8, 8), red);
    comb.position.set(-0.04 + i * 0.04, 0.12 - Math.abs(i - 1) * 0.02, 0);
    head.add(comb);
  }
  for (const side of [-1, 1]) {
    const eye = new Mesh(new SphereGeometry(0.018, 8, 8), new MeshStandardMaterial({ color: "#1d140c" }));
    eye.position.set(0.06, 0.03, side * 0.09);
    head.add(eye);
  }
  body.add(head);
  body.position.y = 0.24;
  group.add(body);
  for (const side of [-1, 1]) {
    const leg = new Mesh(new CylinderGeometry(0.015, 0.015, 0.12, 6), new MeshStandardMaterial({ color: "#ff9a3d" }));
    leg.position.set(0, 0.06, side * 0.08);
    group.add(leg);
  }
  return { group, body, head };
}

export function rainbow(radius: number): Group {
  const g = new Group();
  ["#ff5d5d", "#ff9a3d", "#ffd23f", "#6fd05a", "#4fb0ff", "#5a6aff", "#b38cff"].forEach((color, i) => {
    const band = new Mesh(new TorusGeometry(radius - i * 0.14, 0.07, 8, 64, Math.PI), new MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35, transparent: true, opacity: 0.85, roughness: 0.4 }));
    g.add(band);
  });
  return g;
}
