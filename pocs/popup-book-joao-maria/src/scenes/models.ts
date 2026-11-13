import {
  BoxGeometry,
  type CanvasTexture,
  CatmullRomCurve3,
  CylinderGeometry,
  DodecahedronGeometry,
  ExtrudeGeometry,
  Group,
  LatheGeometry,
  Mesh,
  MeshStandardMaterial,
  Path,
  RepeatWrapping,
  Shape,
  SphereGeometry,
  type Texture,
  TubeGeometry,
  Vector2,
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

export function cauldron(): { group: Group; water: Mesh } {
  const g = new Group();
  const points = [
    new Vector2(0.0, 0.02), new Vector2(0.35, 0.05), new Vector2(0.55, 0.2), new Vector2(0.62, 0.42),
    new Vector2(0.58, 0.62), new Vector2(0.5, 0.72), new Vector2(0.54, 0.76), new Vector2(0.5, 0.78),
  ];
  const pot = new Mesh(new LatheGeometry(points, 32), new MeshStandardMaterial({ color: "#2a2a30", metalness: 0.6, roughness: 0.35, side: 2 }));
  pot.position.y = 0.18;
  const water = new Mesh(new CylinderGeometry(0.5, 0.5, 0.02, 32), new MeshStandardMaterial({ color: "#7fd0f0", emissive: "#2a7fb0", emissiveIntensity: 0.4, roughness: 0.15 }));
  water.position.y = 0.86;
  for (let i = 0; i < 3; i++) {
    const leg = new Mesh(new CylinderGeometry(0.05, 0.03, 0.25, 8), ink);
    const a = (i / 3) * Math.PI * 2;
    leg.position.set(Math.cos(a) * 0.35, 0.12, Math.sin(a) * 0.35);
    g.add(leg);
  }
  g.add(pot, water);
  return { group: g, water };
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

export interface CandyHouse extends HouseParts {
  door: Group;
}

const CANDY = ["#ff5d8f", "#ffd23f", "#7ad7ff", "#8fe36a", "#b38cff", "#ff8a3d"];

function gingerbreadTexture(): CanvasTexture {
  const rand = seeded(8);
  return toTexture(
    paint(256, 256, (ctx, w, h) => {
      ctx.fillStyle = linear(ctx, 0, h, ["#c98a4a", "#a8662e"]);
      ctx.fillRect(0, 0, w, h);
      for (let i = 0; i < 300; i++) {
        ctx.fillStyle = `rgba(90,45,15,${rand() * 0.25})`;
        ctx.fillRect(rand() * w, rand() * h, 3, 3);
      }
      ctx.strokeStyle = "#fffaf0";
      ctx.lineWidth = 9;
      for (const y of [18, h - 18]) {
        ctx.beginPath();
        for (let x = 0; x <= w; x += 32) ctx.arc(x + 16, y, 16, Math.PI, 0, y > h / 2);
        ctx.stroke();
      }
      ctx.lineWidth = 6;
      ctx.beginPath();
      ctx.moveTo(w / 2, 70);
      ctx.bezierCurveTo(w / 2 - 50, 30, w / 2 - 70, 110, w / 2, 150);
      ctx.bezierCurveTo(w / 2 + 70, 110, w / 2 + 50, 30, w / 2, 70);
      ctx.stroke();
    }),
  );
}

function chocolateTexture(): CanvasTexture {
  return toTexture(
    paint(256, 256, (ctx, w, h) => {
      ctx.fillStyle = "#3e1f0e";
      ctx.fillRect(0, 0, w, h);
      for (let y = 0; y < h; y += 64) for (let x = 0; x < w; x += 64) {
        ctx.fillStyle = linear(ctx, y, y + 64, ["#8a4a24", "#5a2f14"]);
        ctx.fillRect(x + 6, y + 6, 52, 52);
        ctx.fillStyle = "rgba(255,220,180,0.25)";
        ctx.fillRect(x + 10, y + 10, 40, 8);
      }
    }),
  );
}

function stripeTexture(color: string): CanvasTexture {
  return toTexture(
    paint(128, 128, (ctx, w, h) => {
      ctx.fillStyle = "#fffaf0";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = color;
      for (let x = -w; x < w * 2; x += 48) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x + 24, 0);
        ctx.lineTo(x + 24 + h, h);
        ctx.lineTo(x + h, h);
        ctx.closePath();
        ctx.fill();
      }
    }),
  );
}

function swirlTexture(a: string, b: string): CanvasTexture {
  return toTexture(
    paint(256, 256, (ctx, w, h) => {
      ctx.fillStyle = a;
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = b;
      ctx.lineWidth = 16;
      ctx.beginPath();
      for (let t = 0; t < Math.PI * 8; t += 0.1) ctx.lineTo(w / 2 + Math.cos(t) * t * 4.8, h / 2 + Math.sin(t) * t * 4.8);
      ctx.stroke();
    }),
  );
}

function glossy(color: string): MeshStandardMaterial {
  return new MeshStandardMaterial({ color, roughness: 0.25, metalness: 0.05 });
}

export function gumdrop(color: string, size = 0.1): Mesh {
  const mesh = new Mesh(new SphereGeometry(size, 16, 10, 0, Math.PI * 2, 0, Math.PI / 2), glossy(color));
  mesh.scale.y = 1.3;
  return mesh;
}

export function lollipop(a: string, b: string, height = 0.9): Group {
  const g = new Group();
  const stick = new Mesh(new CylinderGeometry(0.025, 0.025, height, 8), new MeshStandardMaterial({ color: "#fffaf0" }));
  stick.position.y = height / 2;
  const swirl = swirlTexture(a, b);
  const disc = new Mesh(new CylinderGeometry(0.28, 0.28, 0.08, 32), [glossy(a), new MeshStandardMaterial({ map: swirl, roughness: 0.3 }), new MeshStandardMaterial({ map: swirl, roughness: 0.3 })]);
  disc.rotation.x = Math.PI / 2;
  disc.position.y = height + 0.2;
  g.add(stick, disc);
  return g;
}

export function candyCane(height = 0.9, color = "#e0413a"): Mesh {
  const curve = new CatmullRomCurve3([
    new Vector3(0, 0, 0),
    new Vector3(0, height * 0.7, 0),
    new Vector3(0.05, height * 0.92, 0),
    new Vector3(0.18, height, 0),
    new Vector3(0.3, height * 0.9, 0),
    new Vector3(0.32, height * 0.78, 0),
  ]);
  const texture = stripeTexture(color);
  texture.wrapS = RepeatWrapping;
  texture.wrapT = RepeatWrapping;
  texture.repeat.set(6, 1);
  return new Mesh(new TubeGeometry(curve, 48, 0.055, 12), new MeshStandardMaterial({ map: texture, roughness: 0.3 }));
}

export function candyHouse(): CandyHouse {
  const walls = new Group();
  const ginger = new MeshStandardMaterial({ map: repeat(gingerbreadTexture(), 2, 1), roughness: 0.8 });
  const body = new Mesh(new BoxGeometry(1.8, 1.25, 1.3), ginger);
  body.position.y = 0.625;
  walls.add(body);
  const frame = new Mesh(new BoxGeometry(0.56, 0.86, 0.04), new MeshStandardMaterial({ map: repeat(stripeTexture("#ff5d8f"), 2, 2), roughness: 0.4 }));
  frame.position.set(0, 0.43, 0.66);
  walls.add(frame);
  const door = new Group();
  const panel = new Mesh(new BoxGeometry(0.42, 0.74, 0.05), new MeshStandardMaterial({ map: chocolateTexture(), roughness: 0.5 }));
  panel.position.set(0.21, 0.37, 0);
  const knob = gumdrop("#ffd23f", 0.04);
  knob.rotation.x = Math.PI / 2;
  knob.position.set(0.34, 0.38, 0.03);
  door.add(panel, knob);
  door.position.set(-0.21, 0, 0.69);
  walls.add(door);
  for (const x of [-0.58, 0.58]) {
    const pane = windowPane(0.32, x, 0.78, 0.66);
    walls.add(pane);
  }
  const rand = seeded(4);
  for (let i = 0; i < 18; i++) {
    const side = i % 2 ? 1 : -1;
    const drop = new Mesh(new SphereGeometry(0.045, 10, 8), glossy(CANDY[i % CANDY.length] ?? "#ff5d8f"));
    drop.position.set(-0.85 + rand() * 1.7, 0.15 + rand() * 0.95, side * 0.66);
    if (Math.abs(drop.position.x) < 0.35 && side > 0) continue;
    walls.add(drop);
  }
  const roof = new Group();
  const shape = new Shape();
  shape.moveTo(-1.12, 0);
  shape.lineTo(1.12, 0);
  shape.lineTo(0, 0.95);
  shape.closePath();
  const choco = new Mesh(new ExtrudeGeometry(shape, { depth: 1.55, bevelEnabled: true, bevelSize: 0.05, bevelThickness: 0.05, bevelSegments: 2 }), new MeshStandardMaterial({ map: repeat(chocolateTexture(), 1, 1), roughness: 0.5 }));
  choco.position.set(0, 1.25, -0.775);
  roof.add(choco);
  for (let i = 0; i <= 8; i++) {
    for (const side of [-1, 1]) {
      const icing = new Mesh(new SphereGeometry(0.075, 10, 8), new MeshStandardMaterial({ color: "#fffaf0", roughness: 0.6 }));
      icing.position.set(side * (1.12 - i * 0.14), 1.25 + i * 0.119, 0.84);
      roof.add(icing);
    }
  }
  for (let i = 0; i < 6; i++) {
    const drop = gumdrop(CANDY[i] ?? "#ff5d8f", 0.09);
    drop.position.set(0, 2.2, -0.65 + i * 0.26);
    roof.add(drop);
  }
  const chimney = candyCane(0.75);
  chimney.position.set(0.55, 1.6, -0.2);
  roof.add(chimney);
  const group = new Group();
  group.add(walls, roof);
  return { group, walls, roof, door };
}

export function cage(): Group {
  const g = new Group();
  const iron = new MeshStandardMaterial({ color: "#3a3540", metalness: 0.7, roughness: 0.35 });
  const base = new Mesh(new CylinderGeometry(0.62, 0.66, 0.1, 32), new MeshStandardMaterial({ color: "#6b4220", roughness: 0.8 }));
  base.position.y = 0.05;
  g.add(base);
  const bars = 14;
  for (let i = 0; i < bars; i++) {
    const a = (i / bars) * Math.PI * 2;
    const bar = new Mesh(new CylinderGeometry(0.022, 0.022, 1.35, 8), iron);
    bar.position.set(Math.cos(a) * 0.58, 0.77, Math.sin(a) * 0.58);
    g.add(bar);
  }
  for (const y of [0.12, 1.42]) {
    const ring = new Mesh(new CylinderGeometry(0.6, 0.6, 0.05, 32, 1, true), iron);
    ring.position.y = y;
    g.add(ring);
  }
  const dome = new Mesh(new SphereGeometry(0.6, 24, 10, 0, Math.PI * 2, 0, Math.PI / 2), iron);
  dome.scale.y = 0.45;
  dome.position.y = 1.43;
  const hook = new Mesh(new CylinderGeometry(0.03, 0.03, 0.3, 8), iron);
  hook.position.y = 1.8;
  g.add(dome, hook);
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
}

export function treasureChest(): Chest {
  const g = new Group();
  const wood = new MeshStandardMaterial({ map: repeat(logTexture(), 1, 1), roughness: 0.8 });
  const gold = new MeshStandardMaterial({ color: "#ffc93c", metalness: 0.8, roughness: 0.3 });
  const box = new Mesh(new BoxGeometry(0.8, 0.45, 0.5), wood);
  box.position.y = 0.225;
  g.add(box);
  for (const x of [-0.3, 0.3]) {
    const band = new Mesh(new BoxGeometry(0.06, 0.47, 0.52), gold);
    band.position.set(x, 0.225, 0);
    g.add(band);
  }
  const coins = new Mesh(new SphereGeometry(0.36, 20, 8, 0, Math.PI * 2, 0, Math.PI / 2), gold);
  coins.scale.set(1, 0.45, 0.62);
  coins.position.y = 0.43;
  g.add(coins);
  const rand = seeded(13);
  for (let i = 0; i < 6; i++) {
    const gem = new Mesh(new DodecahedronGeometry(0.05, 0), new MeshStandardMaterial({ color: CANDY[i] ?? "#ff5d8f", emissive: CANDY[i] ?? "#ff5d8f", emissiveIntensity: 0.4, roughness: 0.2, flatShading: true }));
    gem.position.set(-0.25 + rand() * 0.5, 0.5, -0.1 + rand() * 0.2);
    g.add(gem);
  }
  const lid = new Group();
  const top = new Mesh(new CylinderGeometry(0.25, 0.25, 0.8, 20, 1, false, 0, Math.PI), wood);
  top.rotation.z = Math.PI / 2;
  top.rotation.y = Math.PI / 2;
  top.position.set(0, 0, 0.25);
  const lock = new Mesh(new BoxGeometry(0.1, 0.12, 0.04), gold);
  lock.position.set(0, -0.02, 0.51);
  lid.add(top, lock);
  lid.position.set(0, 0.45, -0.25);
  g.add(lid);
  return { group: g, lid };
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

export function crumb(seed: number): Mesh {
  const rand = seeded(seed);
  const mesh = new Mesh(new DodecahedronGeometry(0.06, 0), new MeshStandardMaterial({ color: "#e8b46a", roughness: 0.9, flatShading: true }));
  mesh.scale.set(1 + rand() * 0.4, 0.6, 1 + rand() * 0.3);
  mesh.castShadow = true;
  return mesh;
}

export function glowCap(color: string): Group {
  const g = mushroom();
  g.traverse((child) => {
    if (child instanceof Mesh && child.geometry instanceof SphereGeometry && child.geometry.parameters.radius > 0.1) {
      child.material = new MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.9 });
    }
  });
  return g;
}
