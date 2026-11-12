import { paintPig, paintWolf } from "../art/characters";
import { paintBanner, paintBush, paintButterfly, paintPine, paintTree } from "../art/props";
import { cutoutMesh } from "../engine/popup";
import type { PopupScene } from "../engine/popup";
import { brickHouse, brickStack, cottage, fence, hayBale, logPile, mushroom, rock, stickHouse, strawHouse } from "./houses";
import { type SceneBuilder, bob, createScene, pig, sky, smoke, sparkles, trees } from "./kit";

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

export const cover: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "cover");
  sky(scene, "night");
  trees(scene, [[-3.3, -1.6, 2.4], [3.2, -0.9, 2.6]]);
  const wolf = scene.cutout(paintWolf("sly"), 1.9, 2.6, -1.3);
  const banner = scene.cutout(paintBanner(["The Three", "Little Pigs"], "#e8505b"), 1.75, 0, -0.9, { delay: 0.2 });
  banner.position.y = 1.0;
  const pigs = [0, 1, 2].map((i) =>
    scene.cutout(paintPig({ outfit: ["#3a7bd5", "#e0702c", "#7b4fc9"][i] ?? "#3a7bd5", hat: (["straw", "cap", "bow"] as const)[i] ?? "bow", mood: "joy", wave: i === 1 }), 1.2, -1.5 + i * 1.5, 0.95),
  );
  scene.solid(mushroom(), -2.6, 1.6, 0.6);
  scene.solid(mushroom(), 2.9, 1.4, 0.65);
  scene.solid(rock(0.2, 4), -3.4, 0.9, 0.6);
  sparkles(scene, ctx.assets, [-3.8, 3.8, 0.4, 3.2], ["#ffe27a", "#ffd0f0", "#ffffff"]);
  scene.onUpdate((t) => {
    banner.rotation.z = Math.sin(t * 1.3) * 0.03;
    banner.position.y = 1.0 + Math.sin(t * 1.7) * 0.06;
    wolf.position.x = 0.35 + Math.sin(t * 0.9) * 0.55;
    wolf.rotation.z = Math.sin(t * 0.9) * 0.08;
    pigs.forEach((p, i) => bob(p, 3.2, 0.25, i * 0.9)(t));
  });
  return scene;
};

export const farewell: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "meadow");
  sky(scene, "day");
  const home = cottage();
  scene.solid(home.group, -2.4, -1.1, 0.15, 0.25);
  scene.solid(fence(1.6), -2.3, 0.2, 0.4);
  smoke(scene, -2.85, 2.2, -1.2);
  trees(scene, [[2.8, -1.7, 2.3], [3.5, -0.4, 1.9]]);
  const mother = scene.cutout(paintPig({ outfit: "#e8505b", hat: "kerchief", mood: "smile", wave: true, apron: true }), 1.55, -1.0, 0.5);
  const kids = (["straw", "sticks", "bricks"] as const).map((name, i) => scene.cutout(pig(name, "joy", i !== 1), 1.05, 0.7 + i * 0.95, 0.95 + i * 0.25));
  scene.cutout(paintBush(2), 0.6, -3.4, 1.6);
  butterflies(scene, 3);
  scene.onUpdate((t) => {
    mother.rotation.z = Math.sin(t * 2.2) * 0.05;
    kids.forEach((kid, i) => {
      const walk = (t * 0.25 + i * 0.12) % 1;
      kid.position.x = walk * 1.4;
      bob(kid, 7, 0.12, i)(t);
    });
  });
  return scene;
};

export const straw: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "meadow");
  sky(scene, "day");
  trees(scene, [[-3.2, -1.5, 2.2], [3.4, -1.7, 2.0]]);
  const house = strawHouse();
  scene.solid(house.group, -0.4, -0.7, 0.15);
  scene.solid(hayBale(), -2.4, 0.4, 0.45, 0.4);
  scene.solid(hayBale(), -1.9, 1.2, 0.55, -0.3);
  scene.solid(hayBale(), 2.5, -0.2, 0.4, 0.2);
  const hero = scene.cutout(pig("straw", "joy", true), 1.2, 1.2, 0.8);
  scene.cutout(paintBush(9), 0.55, 3.2, 1.4);
  butterflies(scene, 2);
  sparkles(scene, ctx.assets, [-1.4, 0.6, 0.4, 2.6], ["#ffe27a", "#fff3b0"]);
  scene.onUpdate((t) => {
    bob(hero, 4, 0.3)(t);
    house.roof.rotation.y = Math.sin(t * 0.8) * 0.04;
  });
  return scene;
};

export const sticks: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "meadow");
  sky(scene, "day");
  const pine = paintPine();
  for (const [x, z, h] of [[-3.5, -1.8, 2.3], [-2.7, -1.4, 1.9], [3.4, -1.8, 2.2], [2.8, -1.2, 1.6]] as const) scene.cutout(pine, h, x, z);
  const house = stickHouse();
  scene.solid(house.group, 0.3, -0.8, 0.15, -0.2);
  scene.solid(logPile(), -1.7, 0.7, 0.5, 0.5);
  scene.solid(mushroom(), -2.6, 1.3, 0.6);
  scene.solid(mushroom(), 2.4, 1.6, 0.6);
  scene.solid(rock(0.25, 2), 3.2, 0.6, 0.6);
  const hero = scene.cutout(pig("sticks", "smile"), 1.2, 1.6, 0.6);
  scene.onUpdate((t) => {
    const tap = Math.pow(Math.abs(Math.sin(t * 5)), 6);
    hero.rotation.z = -0.15 + tap * 0.3;
    hero.position.y = tap * 0.08;
  });
  return scene;
};

export const bricks: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "meadow");
  sky(scene, "day");
  trees(scene, [[3.4, -1.6, 2.4]]);
  const house = brickHouse(ctx.assets.bricks);
  scene.solid(house.group, -0.4, -0.9, 0.15);
  scene.solid(brickStack(ctx.assets.bricks), 1.1, 0.9, 0.5);
  scene.solid(brickStack(ctx.assets.bricks), 0.6, 1.3, 0.55, 0.6);
  const hero = scene.cutout(pig("bricks", "smile"), 1.25, 2.0, 0.7);
  const brothers = [scene.cutout(pig("straw", "joy"), 1.0, -2.8, 0.7), scene.cutout(pig("sticks", "joy"), 1.0, -2.0, 1.1)];
  scene.cutout(paintTree(5), 1.9, -3.2, -1.5);
  scene.onUpdate((t) => {
    hero.rotation.z = Math.sin(t * 2) * 0.05;
    brothers.forEach((b, i) => {
      b.rotation.z = Math.sin(t * 9 + i) * 0.12;
      b.position.y = Math.abs(Math.sin(t * 9 + i)) * 0.1;
    });
  });
  return scene;
};
