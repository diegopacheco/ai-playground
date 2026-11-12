import { CanvasTexture, type Texture } from "three";
import { type Ground, paintBackdrop } from "../art/backdrops";
import { toTexture } from "../art/canvas";
import { type PigLook, paintPig } from "../art/characters";
import { paintCloud, paintHills, paintPuff, paintSun, paintTree, paintWindStreak } from "../art/props";
import { Particles, type ParticleOptions } from "../engine/effects";
import { PopupScene } from "../engine/popup";

export interface Assets {
  grass: HTMLImageElement;
  bricks: Texture;
  spark: Texture;
  disc: Texture;
}

export interface SceneContext {
  assets: Assets;
  page: number;
}

export type SceneBuilder = (ctx: SceneContext) => PopupScene;

export const PIGS: Record<"straw" | "sticks" | "bricks", Omit<PigLook, "mood">> = {
  straw: { outfit: "#3a7bd5", hat: "straw" },
  sticks: { outfit: "#e0702c", hat: "cap" },
  bricks: { outfit: "#7b4fc9", hat: "bow" },
};

let puffCache: CanvasTexture | undefined;
export function puffTexture(): CanvasTexture {
  puffCache ??= toTexture(paintPuff());
  return puffCache;
}

let windCache: CanvasTexture | undefined;
export function windTexture(): CanvasTexture {
  windCache ??= toTexture(paintWindStreak());
  return windCache;
}

export function pig(name: keyof typeof PIGS, mood: PigLook["mood"], wave = false): HTMLCanvasElement {
  return paintPig({ ...PIGS[name], mood, wave });
}

export function createScene(ctx: SceneContext, ground: Ground): PopupScene {
  return new PopupScene(paintBackdrop({ ground, page: ctx.page, seed: ctx.page * 17 + 3 }, ctx.assets.grass));
}

export function sky(scene: PopupScene, mood: "day" | "dusk" | "night"): void {
  const palette: Record<typeof mood, [string, string]> = {
    day: ["#7ec8f2", "#d9f1ff"],
    dusk: ["#ff9a76", "#ffd5a1"],
    night: ["#13183d", "#3b2f6b"],
  };
  scene.cutout(paintHills(palette[mood], mood), 3.0, 0, -2.6, { delay: 0 });
  if (mood !== "night") {
    const sun = scene.cutout(paintSun(), 1.1, 2.6, -2.3, { delay: 0.05 });
    sun.position.y = 1.7;
    scene.onUpdate((t) => {
      sun.rotation.z = Math.sin(t * 0.6) * 0.12;
    });
  }
  const clouds = [[-2.4, 2.1, 0.7], [0.6, 2.35, 0.55]] as const;
  const cloudCanvas = paintCloud();
  clouds.forEach(([x, y, h], i) => {
    const cloud = scene.cutout(cloudCanvas, h, x, -2.45, { delay: 0.04 });
    cloud.position.y = y;
    scene.onUpdate((t) => {
      cloud.position.x = Math.sin(t * 0.25 + i * 2) * 0.5;
    });
  });
}

export function trees(scene: PopupScene, spots: [number, number, number][], autumn = false): void {
  spots.forEach(([x, z, h], i) => scene.cutout(paintTree(i * 31 + 7, autumn), h, x, z));
}

export function bob(target: { position: { y: number }; rotation: { z: number } }, speed: number, height: number, phase = 0): (t: number) => void {
  return (t) => {
    target.position.y = Math.abs(Math.sin(t * speed + phase)) * height;
    target.rotation.z = Math.sin(t * speed + phase) * 0.06;
  };
}

export function particles(scene: PopupScene, options: ParticleOptions): Particles {
  const system = new Particles(options);
  for (const sprite of system.sprites) scene.free(sprite);
  scene.onUpdate((_, dt) => system.update(dt));
  return system;
}

export function smoke(scene: PopupScene, x: number, y: number, z: number): Particles {
  return particles(scene, {
    texture: puffTexture(),
    count: 24,
    rate: 5,
    grow: 2.2,
    colors: ["#ffffff", "#e8e4f0", "#d8d2e4"],
    spawn: (p, rand) => {
      p.sprite.position.set(x + (rand() - 0.5) * 0.08, y, z);
      p.velocity.set(0.12 + rand() * 0.1, 0.45 + rand() * 0.2, 0);
      p.life = 2.6 + rand();
      p.size = 0.22 + rand() * 0.1;
    },
  });
}

export function sparkles(scene: PopupScene, assets: Assets, area: [number, number, number, number], colors: string[]): Particles {
  const [x0, x1, y0, y1] = area;
  return particles(scene, {
    texture: assets.spark,
    count: 60,
    rate: 14,
    additive: true,
    colors,
    spawn: (p, rand) => {
      p.sprite.position.set(x0 + rand() * (x1 - x0), y0 + rand() * (y1 - y0), -0.5 + rand() * 2.5);
      p.velocity.set((rand() - 0.5) * 0.2, 0.15 + rand() * 0.2, 0);
      p.life = 1.2 + rand() * 1.5;
      p.size = 0.12 + rand() * 0.18;
    },
  });
}

export function wind(scene: PopupScene, x: number, y: number, z: number, dir: 1 | -1): Particles {
  return particles(scene, {
    texture: windTexture(),
    count: 40,
    rate: 22,
    drag: 0.3,
    spawn: (p, rand) => {
      p.sprite.position.set(x, y + (rand() - 0.5) * 0.7, z + (rand() - 0.5) * 0.8);
      p.velocity.set(dir * (3.2 + rand() * 1.6), (rand() - 0.5) * 0.4, 0);
      p.life = 0.7 + rand() * 0.4;
      p.size = 0.45 + rand() * 0.3;
      p.sprite.material.rotation = dir > 0 ? 0 : Math.PI;
    },
  });
}

export function debris(scene: PopupScene, texture: Texture, x: number, y: number, z: number, dir: 1 | -1): Particles {
  return particles(scene, {
    texture,
    count: 70,
    rate: 45,
    gravity: 1.6,
    spawn: (p, rand) => {
      p.sprite.position.set(x + (rand() - 0.5) * 1.2, y + rand() * 1.4, z + (rand() - 0.5) * 0.8);
      p.velocity.set(dir * (2 + rand() * 2.5), 1 + rand() * 1.8, (rand() - 0.2) * 1.2);
      p.life = 1.2 + rand() * 0.8;
      p.size = 0.16 + rand() * 0.16;
      p.spin = (rand() - 0.5) * 14;
    },
  });
}

export function loop(t: number, length: number): number {
  return ((t % length) + length) % length;
}

export function smooth(t: number, a: number, b: number): number {
  const x = Math.min(Math.max((t - a) / (b - a), 0), 1);
  return x * x * (3 - 2 * x);
}
