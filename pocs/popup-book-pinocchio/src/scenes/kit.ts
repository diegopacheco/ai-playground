import {
  BufferAttribute,
  BufferGeometry,
  type CanvasTexture,
  ConeGeometry,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshStandardMaterial,
  type Texture,
  Vector3,
} from "three";
import { type DropId, paintDrop } from "../art/backdrops";
import { toTexture } from "../art/canvas";
import { paintChip, paintCoin, paintConfetti, paintHeart, paintNote, paintPuff, paintRing, paintStar } from "../art/props";
import { type Figure, type PinocchioLook, paintDonkeyEar, paintPinocchio } from "../art/puppets";
import { Particles, type ParticleOptions } from "../engine/effects";
import { type Cutout, type CutoutOptions, FLOOR_Y, type Mood, PopupScene, cutoutMesh } from "../engine/popup";
import { OPENING_H } from "../engine/theater";

export { loop, smooth } from "../engine/timeline";
export type { Cutout };

export interface Assets {
  spark: Texture;
  disc: Texture;
}

export interface SceneContext {
  assets: Assets;
  page: number;
}

export type SceneBuilder = (ctx: SceneContext) => PopupScene;

export interface Actor {
  mesh: Cutout;
  point: (key: string) => Vector3;
}

const STRING_TOP = FLOOR_Y + OPENING_H + 0.6;
const textures = new Map<string, CanvasTexture>();

export function sprite(key: string, painter: () => HTMLCanvasElement): CanvasTexture {
  let texture = textures.get(key);
  if (!texture) {
    texture = toTexture(painter());
    textures.set(key, texture);
  }
  return texture;
}

export function createScene(drop: DropId, mood: Partial<Mood> = {}): PopupScene {
  const scene = new PopupScene(paintDrop(drop));
  scene.mood = { ...scene.mood, ...mood };
  return scene;
}

export function actor(scene: PopupScene, figure: Figure, height: number, x: number, z: number, options: CutoutOptions = {}): Actor {
  const mesh = scene.cutout(figure.canvas, height, x, z, options);
  const width = (figure.canvas.width / figure.canvas.height) * height;
  const point = (key: string) => {
    const [u, v] = figure.at[key] ?? [0.5, 0];
    return new Vector3((u - 0.5) * width, (1 - v) * height, 0);
  };
  return { mesh, point };
}

export function strings(scene: PopupScene, who: Actor, keys: string[]): void {
  const count = keys.length;
  const positions = new Float32Array(count * 6);
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(positions, 3));
  const material = new LineBasicMaterial({ color: "#f5eedc", transparent: true, opacity: 0.7 });
  const lines = new LineSegments(geometry, material);
  lines.frustumCulled = false;
  scene.free(lines);
  const locals = keys.map((k) => who.point(k));
  const world = new Vector3();
  scene.onUpdate(() => {
    who.mesh.updateWorldMatrix(true, false);
    locals.forEach((local, i) => {
      world.copy(local).applyMatrix4(who.mesh.matrixWorld);
      positions.set([world.x, world.y, world.z, world.x * 0.9 + (i - (count - 1) / 2) * 0.12, STRING_TOP, world.z], i * 6);
    });
    geometry.attributes.position!.needsUpdate = true;
    material.opacity = 0.7 * Math.max(scene.openness * 2 - 1, 0);
  });
}

export interface PinocchioActor extends Actor {
  nose: Mesh<ConeGeometry, MeshStandardMaterial>;
  setNose: (length: number) => void;
  setEars: (growth: number) => void;
}

export function pinocchio(scene: PopupScene, look: PinocchioLook, height: number, x: number, z: number, options: CutoutOptions & { ears?: boolean } = {}): PinocchioActor {
  const who = actor(scene, paintPinocchio(look), height, x, z, options);
  const radius = height * 0.022;
  const geometry = new ConeGeometry(radius, 1, 16).translate(0, 0.5, 0).rotateZ(-Math.PI / 2);
  const nose = new Mesh(geometry, new MeshStandardMaterial({ color: look.real ? "#f6c7a2" : "#d9965a", roughness: 0.6 }));
  nose.castShadow = true;
  nose.position.copy(who.point("nose")).setZ(0.03);
  nose.rotation.y = -0.35;
  who.mesh.add(nose);
  const setNose = (length: number) => {
    nose.scale.set(length * height * 0.12, 1, 1);
  };
  setNose(look.real ? 0.5 : 1);
  const ears: Cutout[] = [];
  if (options.ears) {
    for (const [key, tilt] of [["earL", 0.35], ["earR", -0.35]] as const) {
      const ear = cutoutMesh(paintDonkeyEar(), height * 0.32);
      ear.position.copy(who.point(key)).setZ(-0.01);
      ear.rotation.z = tilt;
      ear.visible = false;
      who.mesh.add(ear);
      ears.push(ear);
    }
  }
  const setEars = (growth: number) => {
    for (const ear of ears) {
      ear.visible = growth > 0.01;
      ear.scale.set(0.6 + growth * 0.4, Math.max(growth, 0.001), 1);
    }
  };
  return { ...who, nose, setNose, setEars };
}

export function dangle(mesh: Cutout, t: number, speed: number, height: number, phase = 0): void {
  mesh.position.y = Math.abs(Math.sin(t * speed + phase)) * height;
  mesh.rotation.z = Math.sin(t * speed * 0.5 + phase) * 0.08;
}

export function pop(mesh: { scale: Vector3; visible: boolean }, amount: number): void {
  const s = Math.max(amount, 0.0001);
  mesh.scale.set(s, s, s);
  mesh.visible = amount > 0.01;
}

export function particles(scene: PopupScene, options: ParticleOptions): Particles {
  const system = new Particles(options);
  for (const item of system.sprites) scene.free(item);
  scene.onUpdate((_, dt) => system.update(dt));
  return system;
}

export function sparkles(scene: PopupScene, assets: Assets, area: [number, number, number, number], colors: string[], rate = 14): Particles {
  const [x0, x1, y0, y1] = area;
  return particles(scene, {
    texture: assets.spark,
    count: 70,
    rate,
    additive: true,
    colors,
    spawn: (p, rand) => {
      p.sprite.position.set(x0 + rand() * (x1 - x0), FLOOR_Y + y0 + rand() * (y1 - y0), -1 + rand() * 2.5);
      p.velocity.set((rand() - 0.5) * 0.2, 0.15 + rand() * 0.2, 0);
      p.life = 1.2 + rand() * 1.5;
      p.size = 0.12 + rand() * 0.18;
    },
  });
}

export function confetti(scene: PopupScene, rate: number): Particles {
  return particles(scene, {
    texture: sprite("confetti", paintConfetti),
    count: 120,
    rate,
    gravity: 0.6,
    drag: 0.8,
    colors: ["#e8505b", "#ffd23f", "#3fa8e8", "#4caf50", "#ff6fa3", "#9b6bff"],
    spawn: (p, rand) => {
      p.sprite.position.set(-4 + rand() * 8, FLOOR_Y + 5.2, -1.5 + rand() * 3);
      p.velocity.set((rand() - 0.5) * 0.6, -0.3 - rand() * 0.4, 0);
      p.life = 5 + rand() * 2;
      p.size = 0.08 + rand() * 0.06;
      p.spin = (rand() - 0.5) * 8;
    },
  });
}

export function coinBurst(scene: PopupScene, x: number, y: number, z: number): Particles {
  return particles(scene, {
    texture: sprite("coin", paintCoin),
    count: 40,
    rate: 0,
    gravity: 4,
    spawn: (p, rand) => {
      p.sprite.position.set(x, FLOOR_Y + y, z);
      p.velocity.set((rand() - 0.5) * 3.2, 2.2 + rand() * 1.8, (rand() - 0.2) * 1.2);
      p.life = 1.4 + rand() * 0.4;
      p.size = 0.2 + rand() * 0.08;
      p.spin = (rand() - 0.5) * 10;
    },
  });
}

export function chips(scene: PopupScene, x: number, y: number, z: number): Particles {
  return particles(scene, {
    texture: sprite("chip", paintChip),
    count: 50,
    rate: 0,
    gravity: 3,
    spawn: (p, rand) => {
      p.sprite.position.set(x, FLOOR_Y + y, z);
      p.velocity.set((rand() - 0.5) * 2.4, 1 + rand() * 1.6, (rand() - 0.3) * 1);
      p.life = 1 + rand() * 0.5;
      p.size = 0.12 + rand() * 0.08;
      p.spin = (rand() - 0.5) * 14;
    },
  });
}

export function notes(scene: PopupScene, x: number, y: number, z: number, colors: string[]): Particles {
  return particles(scene, {
    texture: sprite("note", paintNote),
    count: 24,
    rate: 2.5,
    colors,
    spawn: (p, rand) => {
      p.sprite.position.set(x, FLOOR_Y + y, z);
      p.velocity.set((rand() - 0.5) * 0.6, 0.5 + rand() * 0.3, 0);
      p.life = 2.4 + rand();
      p.size = 0.22 + rand() * 0.1;
      p.spin = (rand() - 0.5) * 1.5;
    },
  });
}

export function bubbles(scene: PopupScene, area: [number, number], rate: number): Particles {
  return particles(scene, {
    texture: sprite("ring", paintRing),
    count: 60,
    rate,
    spawn: (p, rand) => {
      p.sprite.position.set(area[0] + rand() * (area[1] - area[0]), FLOOR_Y + 0.1, -1.5 + rand() * 3);
      p.velocity.set((rand() - 0.5) * 0.2, 0.5 + rand() * 0.5, 0);
      p.life = 3 + rand() * 2;
      p.size = 0.08 + rand() * 0.12;
    },
  });
}

export function hearts(scene: PopupScene, x: number, y: number, z: number): Particles {
  return particles(scene, {
    texture: sprite("heart", paintHeart),
    count: 30,
    rate: 3,
    spawn: (p, rand) => {
      p.sprite.position.set(x + (rand() - 0.5) * 1.2, FLOOR_Y + y, z);
      p.velocity.set((rand() - 0.5) * 0.3, 0.5 + rand() * 0.3, 0);
      p.life = 2.5 + rand();
      p.size = 0.18 + rand() * 0.1;
    },
  });
}

export function puffs(scene: PopupScene, colors: string[], rate: number, spawnAt: () => Vector3): Particles {
  return particles(scene, {
    texture: sprite("puff", paintPuff),
    count: 40,
    rate,
    grow: 2,
    colors,
    spawn: (p, rand) => {
      p.sprite.position.copy(spawnAt());
      p.velocity.set((rand() - 0.5) * 0.4, 0.3 + rand() * 0.3, 0.2);
      p.life = 1.6 + rand();
      p.size = 0.25 + rand() * 0.15;
    },
  });
}

export function stars(scene: PopupScene, color: string, at: () => Vector3): Particles {
  return particles(scene, {
    texture: sprite(`star${color}`, () => paintStar(color)),
    count: 40,
    rate: 0,
    drag: 1.5,
    spawn: (p, rand) => {
      const a = rand() * Math.PI * 2;
      const s = 1.5 + rand() * 2;
      p.sprite.position.copy(at());
      p.velocity.set(Math.cos(a) * s, Math.sin(a) * s, 0.3);
      p.life = 1.2 + rand() * 0.6;
      p.size = 0.15 + rand() * 0.12;
      p.spin = (rand() - 0.5) * 6;
    },
  });
}

export function every(period: number, offset = 0): (t: number, dt: number) => boolean {
  return (t, dt) => {
    const a = Math.floor((t - offset - dt) / period);
    const b = Math.floor((t - offset) / period);
    return b > a;
  };
}
