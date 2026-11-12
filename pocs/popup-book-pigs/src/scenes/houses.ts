import {
  BoxGeometry,
  type CanvasTexture,
  ConeGeometry,
  CylinderGeometry,
  DodecahedronGeometry,
  ExtrudeGeometry,
  Group,
  LatheGeometry,
  Mesh,
  MeshStandardMaterial,
  RepeatWrapping,
  Shape,
  SphereGeometry,
  type Texture,
  Vector2,
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

function tileTexture(): CanvasTexture {
  return toTexture(
    paint(256, 256, (ctx, w, h) => {
      ctx.fillStyle = linear(ctx, 0, h, ["#c8463a", "#9e2f28"]);
      ctx.fillRect(0, 0, w, h);
      for (let y = 0; y < h; y += 32) {
        for (let x = (y / 32) % 2 ? -16 : 0; x < w; x += 32) {
          ctx.beginPath();
          ctx.arc(x + 16, y + 26, 16, 0, Math.PI);
          ctx.fillStyle = "#b53a30";
          ctx.fill();
          ctx.strokeStyle = "#6e1d18";
          ctx.lineWidth = 3;
          ctx.stroke();
        }
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

export function strawHouse(): HouseParts {
  const straw = strawTexture();
  const walls = new Group();
  const body = new Mesh(new CylinderGeometry(0.72, 0.8, 0.95, 28), new MeshStandardMaterial({ map: repeat(straw, 3, 1), roughness: 1 }));
  body.position.y = 0.475;
  walls.add(body, door(0.36, 0.6, 0.78));
  const win = windowPane(0.22, -0.42, 0.6, 0.62);
  win.rotation.y = -0.5;
  walls.add(win);
  const roof = new Group();
  const cone = new Mesh(new ConeGeometry(1.05, 1.0, 28, 1, true), new MeshStandardMaterial({ map: repeat(straw, 4, 1), roughness: 1, side: 2 }));
  cone.position.y = 1.4;
  const tuft = new Mesh(new ConeGeometry(0.12, 0.3, 12), new MeshStandardMaterial({ color: "#c9962e" }));
  tuft.position.y = 2.0;
  roof.add(cone, tuft);
  const group = new Group();
  group.add(walls, roof);
  return { group, walls, roof };
}

export function stickHouse(): HouseParts {
  const logs = logTexture();
  const wood = new MeshStandardMaterial({ map: repeat(logs, 2, 1), roughness: 0.9 });
  const walls = new Group();
  const body = new Mesh(new BoxGeometry(1.4, 1.0, 1.1), wood);
  body.position.y = 0.5;
  walls.add(body, door(0.34, 0.62, 0.56, "#4a2a12"), windowPane(0.24, 0.42, 0.62, 0.56));
  const roof = new Group();
  const pyramid = new Mesh(new ConeGeometry(1.15, 0.85, 4), new MeshStandardMaterial({ map: repeat(logs, 3, 1), roughness: 0.9 }));
  pyramid.rotation.y = Math.PI / 4;
  pyramid.scale.z = 0.8;
  pyramid.position.y = 1.42;
  roof.add(pyramid);
  for (let i = 0; i < 6; i++) {
    const stick = new Mesh(new CylinderGeometry(0.025, 0.025, 0.5, 6), new MeshStandardMaterial({ color: "#6b4220" }));
    stick.position.set(-0.3 + i * 0.12, 1.9, 0);
    stick.rotation.z = (i - 2.5) * 0.25;
    roof.add(stick);
  }
  const group = new Group();
  group.add(walls, roof);
  return { group, walls, roof };
}

export function brickHouse(bricks: Texture): HouseParts {
  const brick = new MeshStandardMaterial({ map: repeat(bricks, 2, 1.4), roughness: 0.85 });
  const walls = new Group();
  const body = new Mesh(new BoxGeometry(1.8, 1.3, 1.3), brick);
  body.position.y = 0.65;
  const step = new Mesh(new BoxGeometry(0.6, 0.08, 0.25), new MeshStandardMaterial({ color: "#9a9a9a" }));
  step.position.set(0, 0.04, 0.75);
  walls.add(body, step, door(0.4, 0.72, 0.66), windowPane(0.3, -0.55, 0.8, 0.66), windowPane(0.3, 0.55, 0.8, 0.66));
  const roof = new Group();
  const shape = new Shape();
  shape.moveTo(-1.05, 0);
  shape.lineTo(1.05, 0);
  shape.lineTo(0, 0.85);
  shape.closePath();
  const prism = new Mesh(new ExtrudeGeometry(shape, { depth: 1.5, bevelEnabled: false }), new MeshStandardMaterial({ map: repeat(tileTexture(), 1, 1), roughness: 0.7 }));
  prism.position.set(0, 1.3, -0.75);
  const chimney = new Mesh(new BoxGeometry(0.3, 0.7, 0.3), brick);
  chimney.position.set(0.55, 1.85, -0.2);
  const cap = new Mesh(new BoxGeometry(0.38, 0.06, 0.38), ink);
  cap.position.set(0.55, 2.22, -0.2);
  roof.add(prism, chimney, cap);
  const group = new Group();
  group.add(walls, roof);
  return { group, walls, roof };
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

export function hayBale(): Group {
  const mesh = new Mesh(new CylinderGeometry(0.28, 0.28, 0.5, 20), new MeshStandardMaterial({ map: repeat(strawTexture(), 2, 1), roughness: 1 }));
  mesh.rotation.z = Math.PI / 2;
  mesh.position.y = 0.28;
  const g = new Group();
  g.add(mesh);
  return g;
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

export function brickStack(bricks: Texture): Group {
  const g = new Group();
  const material = new MeshStandardMaterial({ map: repeat(bricks, 0.25, 0.12), color: "#ffffff", roughness: 0.85 });
  for (let row = 0; row < 4; row++) {
    for (let i = 0; i < 4 - row; i++) {
      const b = new Mesh(new BoxGeometry(0.22, 0.1, 0.1), material);
      b.position.set(-0.17 + i * 0.23 + row * 0.115, 0.05 + row * 0.1, 0);
      g.add(b);
    }
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

export function fireplace(bricks: Texture): Group {
  const g = new Group();
  const brick = new MeshStandardMaterial({ map: repeat(bricks, 1.5, 1.5), roughness: 0.85 });
  const back = new Mesh(new BoxGeometry(2.6, 2.4, 0.3), brick);
  back.position.set(0, 1.2, -0.45);
  const sideL = new Mesh(new BoxGeometry(0.4, 1.6, 0.9), brick);
  sideL.position.set(-1.1, 0.8, 0);
  const sideR = sideL.clone();
  sideR.position.x = 1.1;
  const mantel = new Mesh(new BoxGeometry(2.8, 0.18, 1.05), new MeshStandardMaterial({ color: "#6b4220", roughness: 0.6 }));
  mantel.position.set(0, 1.68, 0.02);
  const top = new Mesh(new BoxGeometry(2.6, 0.7, 0.9), brick);
  top.position.set(0, 2.1, 0);
  const flue = new Mesh(new BoxGeometry(1.0, 1.4, 0.7), brick);
  flue.position.set(0, 3.1, -0.1);
  const soot = new Mesh(new BoxGeometry(1.8, 1.4, 0.05), new MeshStandardMaterial({ color: "#1c1210", roughness: 1 }));
  soot.position.set(0, 0.8, -0.28);
  g.add(back, sideL, sideR, mantel, top, flue, soot);
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
