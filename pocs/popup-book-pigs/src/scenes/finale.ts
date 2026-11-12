import { PointLight } from "three";
import { toTexture } from "../art/canvas";
import { paintWolf } from "../art/characters";
import { paintBanner, paintBush, paintFlame, paintHeart, paintMoon, paintWall } from "../art/props";
import { cutoutMesh } from "../engine/popup";
import { brickHouse, cauldron, fireplace, logsFire, mushroom } from "./houses";
import { type SceneBuilder, bob, createScene, loop, particles, pig, puffTexture, sky, smooth, sparkles, trees } from "./kit";

export const roof: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "dusk");
  sky(scene, "night");
  const moon = scene.cutout(paintMoon(), 0.9, -2.7, -2.3, { delay: 0.05 });
  moon.position.y = 1.8;
  trees(scene, [[-3.3, -1.0, 2.0], [3.5, -1.3, 2.1]], true);
  const house = brickHouse(ctx.assets.bricks);
  house.group.scale.setScalar(1.3);
  scene.solid(house.group, 0.2, -0.6, 0.15);
  const wolf = scene.cutout(paintWolf("sly"), 1.15, 0.2, -0.6, { delay: 0.5 });
  const plotter = scene.cutout(pig("bricks", "smile"), 1.1, 2.6, 0.9);
  scene.cutout(paintBush(12), 0.55, -2.2, 1.5);
  sparkles(scene, ctx.assets, [-3.8, 3.8, 1.5, 3.4], ["#fff7c2", "#ffffff"]);
  const chimneyX = 0.55 * 1.3;
  scene.onUpdate((t) => {
    const c = loop(t, 7);
    const walk = smooth(c, 0.3, 3.6);
    const drop = smooth(c, 4.2, 5.0);
    const back = smooth(c, 6.4, 7);
    wolf.position.x = -1.1 + walk * (chimneyX + 1.1) - back * walk * (chimneyX + 1.1);
    wolf.position.y = 2.72 + Math.abs(Math.sin(t * 6)) * 0.08 * (1 - drop) + (c > 3.6 && c < 4.2 ? Math.sin(((c - 3.6) / 0.6) * Math.PI) * 0.45 : 0) - drop * 1.6;
    wolf.rotation.z = Math.sin(t * 6) * 0.06;
    wolf.visible = drop < 0.95 || back > 0.5;
    plotter.rotation.z = Math.sin(t * 2.5) * 0.08;
  });
  return scene;
};

export const pot: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "interior");
  scene.cutout(paintWall(), 3.3, 0, -2.75, { delay: 0 });
  scene.solid(fireplace(ctx.assets.bricks), 0, -1.75, 0.1);
  scene.solid(logsFire(), 0, -1.5, 0.3);
  const kettle = cauldron();
  kettle.group.scale.setScalar(0.95);
  scene.solid(kettle.group, 0, -1.45, 0.35);
  const flameCanvas = paintFlame();
  const flames = [-0.55, 0, 0.5].map((x) => {
    const flame = cutoutMesh(flameCanvas, 0.6);
    flame.castShadow = false;
    flame.material.emissive.set("#ff7a1a");
    flame.material.emissiveMap = flame.material.map;
    flame.material.emissiveIntensity = 1.5;
    flame.position.set(x, 0.36, -1.15);
    scene.free(flame);
    return flame;
  });
  const fireLight = new PointLight("#ff8a3a", 18, 7, 1.6);
  fireLight.position.set(0, 1.1, -0.6);
  scene.free(fireLight);
  const faces = { sly: toTexture(paintWolf("puff")), ouch: toTexture(paintWolf("ouch")) };
  const wolf = cutoutMesh(paintWolf("puff"), 1.35);
  wolf.castShadow = false;
  wolf.position.set(0, 4, -1.42);
  scene.free(wolf);
  const steam = particles(scene, {
    texture: puffTexture(),
    count: 40,
    rate: 9,
    grow: 2.5,
    colors: ["#ffffff", "#f0f6ff"],
    spawn: (p, rand) => {
      p.sprite.position.set((rand() - 0.5) * 0.7, 1.3, -1.4 + (rand() - 0.5) * 0.4);
      p.velocity.set((rand() - 0.5) * 0.3, 0.7 + rand() * 0.4, 0.15);
      p.life = 1.8 + rand();
      p.size = 0.25;
    },
  });
  const splash = particles(scene, {
    texture: ctx.assets.disc,
    count: 70,
    rate: 0,
    gravity: 6,
    colors: ["#9fe3ff", "#d9f6ff", "#5bbbe8"],
    spawn: (p, rand) => {
      const a = rand() * Math.PI * 2;
      p.sprite.position.set(Math.cos(a) * 0.3, 1.25, -1.4 + Math.sin(a) * 0.3);
      p.velocity.set(Math.cos(a) * (1 + rand() * 1.5), 2.5 + rand() * 2.5, Math.sin(a) * (1 + rand()) + 0.5);
      p.life = 1 + rand() * 0.5;
      p.size = 0.08 + rand() * 0.08;
    },
  });
  splash.emitting = false;
  const pigs = [scene.cutout(pig("straw", "scared"), 1.0, -2.7, 0.7), scene.cutout(pig("sticks", "scared"), 1.0, -1.9, 1.2), scene.cutout(pig("bricks", "joy"), 1.15, 2.4, 0.9)];
  let splashed = false;
  scene.onUpdate((t) => {
    flames.forEach((f, i) => {
      f.scale.set(1 + Math.sin(t * 13 + i * 2) * 0.12, 1 + Math.sin(t * 17 + i) * 0.22, 1);
    });
    fireLight.intensity = 16 + Math.sin(t * 21) * 3 + Math.sin(t * 7) * 2;
    kettle.water.position.y = 0.86 + Math.sin(t * 12) * 0.012;
    const c = loop(t, 6);
    const fall = smooth(c, 0.5, 1.4);
    const shoot = smooth(c, 2.6, 3.6);
    wolf.position.y = 4.2 - fall * 3.55 + shoot * 4.5;
    wolf.position.x = shoot * 1.6;
    wolf.rotation.z = shoot * 2.6;
    wolf.material.map = c > 1.4 ? faces.ouch : faces.sly;
    wolf.visible = c > 0.4 && shoot < 0.99;
    if (c > 1.35 && c < 1.6 && !splashed) {
      splash.burst(60);
      splashed = true;
    }
    if (c < 1) splashed = false;
    steam.emitting = true;
    pigs.forEach((p, i) => bob(p, i === 2 ? 5 : 20, i === 2 ? 0.2 : 0.03, i)(t));
  });
  return scene;
};

export const theEnd: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "meadow");
  sky(scene, "day");
  trees(scene, [[-3.3, -1.5, 2.3], [3.4, -1.5, 2.3]]);
  const banner = scene.cutout(paintBanner(["The End"], "#e8505b"), 1.05, 0, -1.9, { delay: 0.05 });
  banner.position.y = 2.15;
  const house = brickHouse(ctx.assets.bricks);
  scene.solid(house.group, 0, -0.9, 0.2);
  const dancers = (["straw", "sticks", "bricks"] as const).map((name, i) => scene.cutout(pig(name, "joy", true), 1.15, -1.5 + i * 1.5, 1.0));
  scene.solid(mushroom(), -2.6, 1.5, 0.6);
  scene.solid(mushroom(), 2.7, 1.6, 0.6);
  particles(scene, {
    texture: toTexture(paintHeart()),
    count: 30,
    rate: 4,
    spawn: (p, rand) => {
      p.sprite.position.set(-2 + rand() * 4, 0.8, 1 + rand() * 0.4);
      p.velocity.set((rand() - 0.5) * 0.3, 0.6 + rand() * 0.4, 0);
      p.life = 2.5 + rand();
      p.size = 0.16 + rand() * 0.1;
    },
  });
  let origin = { x: 0, y: 3 };
  const fireworks = particles(scene, {
    texture: ctx.assets.spark,
    count: 160,
    rate: 0,
    additive: true,
    gravity: 0.8,
    drag: 1.2,
    colors: ["#ff5d8f", "#ffd23f", "#7ad7ff", "#b6ff7a", "#ffffff"],
    spawn: (p, rand) => {
      const a = rand() * Math.PI * 2;
      const speed = 1.4 + rand() * 0.6;
      p.sprite.position.set(origin.x, origin.y, -1.5);
      p.velocity.set(Math.cos(a) * speed, Math.sin(a) * speed, 0);
      p.life = 1.3 + rand() * 0.4;
      p.size = 0.22;
    },
  });
  fireworks.emitting = false;
  let next = 0.8;
  scene.onUpdate((t) => {
    if (t > next) {
      origin = { x: -2.8 + Math.random() * 5.6, y: 2.6 + Math.random() * 0.8 };
      fireworks.burst(50);
      next = t + 1.1;
    }
    banner.rotation.z = Math.sin(t * 1.4) * 0.03;
    dancers.forEach((d, i) => {
      d.position.y = Math.abs(Math.sin(t * 5 + i)) * 0.35;
      d.rotation.z = Math.sin(t * 5 + i) * 0.18;
    });
  });
  return scene;
};
