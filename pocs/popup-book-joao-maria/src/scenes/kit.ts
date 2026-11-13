import { CanvasTexture, type Mesh, type MeshStandardMaterial, type PlaneGeometry, type Texture } from "three";
import { type Ground, paintBackdrop } from "../art/backdrops";
import { toTexture } from "../art/canvas";
import { type ChildLook, paintChild } from "../art/characters";
import { paintButterfly, paintCloud, paintHills, paintPine, paintPuff, paintSun, paintTree } from "../art/props";
import { Particles, type ParticleOptions } from "../engine/effects";
import { PopupScene, cutoutMesh } from "../engine/popup";

export { smooth } from "./timelines";

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

export type Cutout = Mesh<PlaneGeometry, MeshStandardMaterial>;

let puffCache: CanvasTexture | undefined;
export function puffTexture(): CanvasTexture {
  puffCache ??= toTexture(paintPuff());
  return puffCache;
}

export function child(name: ChildLook["name"], mood: ChildLook["mood"], extra: Partial<ChildLook> = {}): HTMLCanvasElement {
  return paintChild({ name, mood, ...extra });
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
    if (mood === "night") cloud.material.color.set("#8a8ab8");
    scene.onUpdate((t) => {
      cloud.position.x = Math.sin(t * 0.25 + i * 2) * 0.5;
    });
  });
}

export function trees(scene: PopupScene, spots: [number, number, number][], autumn = false): void {
  spots.forEach(([x, z, h], i) => scene.cutout(paintTree(i * 31 + 7, autumn), h, x, z));
}

export function pines(scene: PopupScene, spots: [number, number, number][]): void {
  const pine = paintPine();
  for (const [x, z, h] of spots) scene.cutout(pine, h, x, z);
}

export function bob(target: { position: { y: number }; rotation: { z: number } }, speed: number, height: number, phase = 0): (t: number) => void {
  return (t) => {
    target.position.y = Math.abs(Math.sin(t * speed + phase)) * height;
    target.rotation.z = Math.sin(t * speed + phase) * 0.06;
  };
}

export function pop(mesh: Cutout, amount: number): void {
  const s = Math.max(amount, 0.0001);
  mesh.scale.set(s, s, s);
  mesh.visible = amount > 0.01;
}

export function faces(canvases: Record<string, HTMLCanvasElement>): Record<string, Texture> {
  return Object.fromEntries(Object.entries(canvases).map(([k, c]) => [k, toTexture(c)]));
}

export function setFace(mesh: Cutout, textures: Record<string, Texture>, key: string): void {
  const face = textures[key];
  if (face && mesh.material.map !== face) {
    mesh.material.map = face;
    mesh.material.needsUpdate = true;
  }
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

export function fireflies(scene: PopupScene, assets: Assets): Particles {
  return particles(scene, {
    texture: assets.spark,
    count: 50,
    rate: 9,
    additive: true,
    colors: ["#d8ff7a", "#fff27a", "#9dff9a"],
    spawn: (p, rand) => {
      p.sprite.position.set(-3.6 + rand() * 7.2, 0.4 + rand() * 1.8, -1.5 + rand() * 3.2);
      p.velocity.set((rand() - 0.5) * 0.35, (rand() - 0.5) * 0.25, (rand() - 0.5) * 0.2);
      p.life = 2 + rand() * 2;
      p.size = 0.12 + rand() * 0.1;
    },
  });
}

export function butterflies(scene: PopupScene, count: number): void {
  const colors = ["#ff8ad8", "#ffd23f", "#7ad7ff", "#b38cff"];
  for (let i = 0; i < count; i++) {
    const wing = cutoutMesh(paintButterfly(colors[i % colors.length] ?? "#ff8ad8"), 0.28);
    wing.castShadow = false;
    scene.free(wing);
    const cx = -2.5 + i * 1.7;
    scene.onUpdate((t) => {
      const a = t * 0.7 + i * 2.1;
      wing.position.set(cx + Math.sin(a) * 0.9, 1.4 + Math.sin(a * 2.3) * 0.35 + 0.34, 0.6 + Math.cos(a) * 0.7);
      wing.scale.x = 0.25 + Math.abs(Math.sin(t * 14 + i)) * 0.75;
    });
  }
}

export function loop(t: number, length: number): number {
  return ((t % length) + length) % length;
}
