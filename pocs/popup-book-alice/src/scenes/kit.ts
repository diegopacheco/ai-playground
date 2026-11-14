import {
  BoxGeometry,
  CanvasTexture,
  CylinderGeometry,
  Group,
  LatheGeometry,
  Mesh,
  MeshPhysicalMaterial,
  MeshStandardMaterial,
  PlaneGeometry,
  SphereGeometry,
  type Texture,
  TorusGeometry,
  Vector2,
} from "three";
import { heart, spade } from "../art/characters";
import { paint, toTexture } from "../art/canvas";
import type { Tunnel } from "../engine/tunnel";

export interface Assets {
  spark: Texture;
  disc: Texture;
}

export type SceneBuilder = (z: number, assets: Assets) => Tunnel;

export const FLOOR = -3.2;

function lathe(points: number[][], material: MeshStandardMaterial, segments = 32): Mesh {
  return new Mesh(new LatheGeometry(points.map(([x, y]) => new Vector2(x, y)), segments), material);
}

const porcelain = new MeshStandardMaterial({ color: "#fdf6ec", roughness: 0.35 });
const gold = new MeshStandardMaterial({ color: "#e0b040", roughness: 0.3, metalness: 0.85 });

function labelTexture(text: string, bg: string, color: string, w = 512, h = 160): CanvasTexture {
  return toTexture(
    paint(w, h, (ctx) => {
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = color;
      ctx.lineWidth = 8;
      ctx.strokeRect(10, 10, w - 20, h - 20);
      ctx.fillStyle = color;
      ctx.font = `700 ${Math.round(h * 0.42)}px 'IM Fell English SC'`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, w / 2, h / 2 + 4, w - 40);
    }),
  );
}

export function teacup(color: string): Group {
  const group = new Group();
  const cup = lathe(
    [
      [0, 0],
      [0.09, 0],
      [0.1, 0.02],
      [0.17, 0.16],
      [0.18, 0.2],
      [0.16, 0.2],
      [0.15, 0.17],
      [0, 0.16],
    ],
    new MeshStandardMaterial({ color, roughness: 0.35 }),
  );
  const handle = new Mesh(new TorusGeometry(0.06, 0.016, 8, 16), porcelain);
  handle.position.set(0.18, 0.11, 0);
  const saucer = lathe(
    [
      [0, 0],
      [0.26, 0.01],
      [0.28, 0.03],
      [0, 0.02],
    ],
    porcelain,
  );
  group.add(saucer, cup, handle);
  return group;
}

export function teapot(): { group: Group; lid: Group } {
  const group = new Group();
  const body = lathe(
    [
      [0, 0],
      [0.25, 0],
      [0.38, 0.12],
      [0.42, 0.3],
      [0.36, 0.48],
      [0.2, 0.56],
      [0, 0.56],
    ],
    new MeshStandardMaterial({ color: "#5aa0d8", roughness: 0.3 }),
  );
  const spout = new Mesh(new CylinderGeometry(0.035, 0.07, 0.4, 12), porcelain);
  spout.position.set(0.48, 0.38, 0);
  spout.rotation.z = -0.8;
  const handle = new Mesh(new TorusGeometry(0.15, 0.03, 8, 20, Math.PI * 1.3), porcelain);
  handle.position.set(-0.42, 0.3, 0);
  handle.rotation.z = Math.PI * 0.35;
  const lid = new Group();
  const cap = lathe(
    [
      [0, 0],
      [0.22, 0],
      [0.16, 0.08],
      [0.04, 0.12],
      [0, 0.16],
    ],
    porcelain,
  );
  const knob = new Mesh(new SphereGeometry(0.05, 12, 8), gold);
  knob.position.y = 0.17;
  lid.add(cap, knob);
  lid.position.y = 0.55;
  group.add(body, spout, handle, lid);
  return { group, lid };
}

export function bottle(): Group {
  const group = new Group();
  const glass = new MeshPhysicalMaterial({ color: "#7fd1c8", roughness: 0.08, transmission: 0.6, thickness: 0.3, transparent: true, opacity: 0.85 });
  group.add(
    lathe(
      [
        [0, 0],
        [0.16, 0],
        [0.18, 0.04],
        [0.18, 0.36],
        [0.07, 0.5],
        [0.06, 0.66],
        [0, 0.66],
      ],
      glass as unknown as MeshStandardMaterial,
    ),
  );
  const cork = new Mesh(new CylinderGeometry(0.06, 0.055, 0.1, 12), new MeshStandardMaterial({ color: "#b5835a" }));
  cork.position.y = 0.7;
  const label = new Mesh(new PlaneGeometry(0.34, 0.14), new MeshStandardMaterial({ map: labelTexture("DRINK ME", "#fffaf0", "#3b2416") }));
  label.position.set(0, 0.22, 0.185);
  group.add(cork, label);
  return group;
}

export function cake(): Group {
  const group = new Group();
  const box = new Mesh(new BoxGeometry(0.9, 0.36, 0.7), new MeshPhysicalMaterial({ color: "#d8f0ff", transmission: 0.5, roughness: 0.05, transparent: true, opacity: 0.55 }));
  box.position.y = 0.18;
  const sponge = new Mesh(new CylinderGeometry(0.26, 0.26, 0.2, 28), new MeshStandardMaterial({ color: "#e8b46a", roughness: 0.8 }));
  sponge.position.y = 0.46;
  const top = new Mesh(new CylinderGeometry(0.265, 0.265, 0.03, 28), [
    new MeshStandardMaterial({ color: "#fff1f5" }),
    new MeshStandardMaterial({ map: labelTexture("EAT ME", "#fff1f5", "#8a2b2b", 256, 256) }),
    new MeshStandardMaterial({ color: "#fff1f5" }),
  ]);
  top.position.y = 0.57;
  group.add(box, sponge, top);
  return group;
}

export function mushroom(radius: number, height: number, capColor: string): Group {
  const group = new Group();
  const stem = lathe(
    [
      [0, 0],
      [radius * 0.32, 0],
      [radius * 0.24, height * 0.5],
      [radius * 0.26, height],
      [0, height],
    ],
    new MeshStandardMaterial({ color: "#f3e6cc", roughness: 0.8 }),
  );
  const cap = lathe(
    [
      [0, height + radius * 0.55],
      [radius * 0.5, height + radius * 0.48],
      [radius * 0.9, height + radius * 0.25],
      [radius, height + radius * 0.02],
      [radius * 0.82, height - radius * 0.04],
      [0, height],
    ],
    new MeshStandardMaterial({ color: capColor, roughness: 0.6 }),
    40,
  );
  group.add(stem, cap);
  const spot = new MeshStandardMaterial({ color: "#fffaf0", roughness: 0.6 });
  for (let i = 0; i < 9; i++) {
    const a = (i / 9) * Math.PI * 2;
    const r = radius * (i % 2 === 0 ? 0.55 : 0.78);
    const s = new Mesh(new SphereGeometry(radius * 0.09, 10, 6), spot);
    s.scale.y = 0.4;
    s.position.set(Math.cos(a) * r, height + radius * (i % 2 === 0 ? 0.44 : 0.28), Math.sin(a) * r);
    group.add(s);
  }
  return group;
}

export function clock(): Group {
  const face = toTexture(
    paint(256, 256, (ctx) => {
      ctx.fillStyle = "#fffaf0";
      ctx.fillRect(0, 0, 256, 256);
      ctx.fillStyle = "#3b2416";
      ctx.font = "700 30px 'IM Fell English'";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const numerals = ["XII", "III", "VI", "IX"];
      numerals.forEach((n, i) => {
        const a = (i / 4) * Math.PI * 2 - Math.PI / 2;
        ctx.fillText(n, 128 + Math.cos(a) * 92, 128 + Math.sin(a) * 92);
      });
      ctx.strokeStyle = "#3b2416";
      ctx.lineWidth = 8;
      ctx.beginPath();
      ctx.moveTo(128, 128);
      ctx.lineTo(128, 60);
      ctx.moveTo(128, 128);
      ctx.lineTo(176, 140);
      ctx.stroke();
    }),
  );
  const group = new Group();
  const rim = new Mesh(new CylinderGeometry(0.4, 0.4, 0.12, 32), [gold, new MeshStandardMaterial({ map: face }), gold]);
  rim.rotation.x = Math.PI / 2;
  group.add(rim);
  return group;
}

export function jar(): Group {
  const group = new Group();
  group.add(
    lathe(
      [
        [0, 0],
        [0.22, 0],
        [0.24, 0.1],
        [0.24, 0.42],
        [0.2, 0.48],
        [0, 0.48],
      ],
      new MeshStandardMaterial({ color: "#f08c00", roughness: 0.3 }),
    ),
  );
  const label = new Mesh(new PlaneGeometry(0.38, 0.2), new MeshStandardMaterial({ map: labelTexture("MARMALADE", "#fffaf0", "#8a2b2b") }));
  label.position.set(0, 0.24, 0.245);
  group.add(label);
  return group;
}

export function book(color: string): Mesh {
  return new Mesh(new BoxGeometry(0.5, 0.12, 0.7), [
    new MeshStandardMaterial({ color: "#fffaf0" }),
    new MeshStandardMaterial({ color }),
    new MeshStandardMaterial({ color }),
    new MeshStandardMaterial({ color }),
    new MeshStandardMaterial({ color: "#fffaf0" }),
    new MeshStandardMaterial({ color: "#fffaf0" }),
  ]);
}

let backs: MeshStandardMaterial | undefined;

function cardBacks(): MeshStandardMaterial {
  backs ??= new MeshStandardMaterial({
  map: toTexture(
    paint(160, 224, (ctx) => {
      ctx.fillStyle = "#b03030";
      ctx.fillRect(0, 0, 160, 224);
      ctx.strokeStyle = "#fffaf0";
      ctx.lineWidth = 6;
      ctx.strokeRect(10, 10, 140, 204);
      for (let y = 30; y < 210; y += 24) for (let x = 26; x < 150; x += 24) heart(ctx, x, y, 6, "rgba(255,240,230,0.6)");
    }),
  ),
  roughness: 0.6,
  });
  return backs;
}

export function card(index: number): Mesh {
  const red = index % 2 === 0;
  const face = new MeshStandardMaterial({
    map: toTexture(
      paint(160, 224, (ctx) => {
        ctx.fillStyle = "#fffdf6";
        ctx.fillRect(0, 0, 160, 224);
        ctx.strokeStyle = "#3b2416";
        ctx.lineWidth = 4;
        ctx.strokeRect(4, 4, 152, 216);
        if (red) heart(ctx, 80, 112, 34, "#e03131");
        else spade(ctx, 80, 104, 32, "#2b2b2b");
        ctx.fillStyle = red ? "#e03131" : "#2b2b2b";
        ctx.font = "700 34px 'IM Fell English'";
        ctx.fillText(String((index % 9) + 2), 14, 40);
      }),
    ),
    roughness: 0.6,
  });
  const back = cardBacks();
  return new Mesh(new BoxGeometry(0.5, 0.7, 0.01), [back, back, back, back, face, back]);
}

export function glassTable(): Group {
  const group = new Group();
  const top = new Mesh(new BoxGeometry(1.8, 0.06, 1.1), new MeshPhysicalMaterial({ color: "#e6f6ff", transmission: 0.7, roughness: 0.05, transparent: true, opacity: 0.6 }));
  top.position.y = 1.6;
  group.add(top);
  for (const [x, z] of [
    [-0.8, -0.45],
    [0.8, -0.45],
    [-0.8, 0.45],
    [0.8, 0.45],
  ] as const) {
    const leg = new Mesh(new CylinderGeometry(0.04, 0.05, 1.6, 10), gold);
    leg.position.set(x, 0.8, z);
    group.add(leg);
  }
  const key = new Group();
  const ring = new Mesh(new TorusGeometry(0.07, 0.02, 8, 16), gold);
  const shaft = new Mesh(new BoxGeometry(0.28, 0.025, 0.025), gold);
  shaft.position.x = 0.2;
  key.add(ring, shaft);
  key.rotation.x = -Math.PI / 2;
  key.position.set(0.45, 1.65, 0.2);
  group.add(key);
  return group;
}
