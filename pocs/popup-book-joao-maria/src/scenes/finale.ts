import { toTexture } from "../art/canvas";
import { paintFather, paintWitch } from "../art/characters";
import { paintBanner, paintCoin, paintHeart, paintKitchenWall } from "../art/props";
import { cage, cottage, fence, mushroom, oven, treasureChest } from "./models";
import { type SceneBuilder, child, createScene, faces, loop, particles, puffTexture, setFace, sky, smoke, smooth, trees } from "./kit";
import { DOOR_SHUT, OVEN_LOOP, doorClose, mariaRun, push, witchLean, witchShowing } from "./timelines";
import { fireLight, flames } from "./witch";

export const ovenScene: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "interior");
  scene.cutout(paintKitchenWall(), 3.3, 0, -2.75, { delay: 0 });
  const stove = oven(ctx.assets.bricks);
  scene.solid(stove.group, 0.6, -1.4, 0.1);
  const fire = flames(scene, [[0.35, 0.36, -1.15], [0.6, 0.36, -1.1], [0.85, 0.36, -1.15]], 0.55);
  const light = fireLight(scene, 0.6, 0.9, -0.5);
  scene.solid(cage(), -2.5, -0.5, 0.2);
  const joaoLooks = faces({ scared: child("joao", "scared"), joy: child("joao", "joy", { wave: true }) });
  const joao = scene.cutout(child("joao", "scared"), 1.0, -2.5, -0.5, { delay: 0.25 });
  const witchLooks = faces({ wicked: paintWitch("wicked"), falling: paintWitch("falling") });
  const witch = scene.cutout(paintWitch("wicked"), 1.5, 0.6, -0.05);
  const mariaLooks = faces({ run: child("maria", "scared"), joy: child("maria", "joy", { wave: true }) });
  const maria = scene.cutout(child("maria", "scared"), 1.1, 0.6, 0.55);
  const sparks = particles(scene, {
    texture: ctx.assets.spark,
    count: 80,
    rate: 0,
    additive: true,
    gravity: -0.4,
    colors: ["#ffd23f", "#ff8a3d", "#fff3b0"],
    spawn: (p, rand) => {
      p.sprite.position.set(1.15 + (rand() - 0.5) * 0.3, 3.0, -1.5);
      p.velocity.set((rand() - 0.5) * 1.2, 0.8 + rand() * 1.4, (rand() - 0.5) * 0.4);
      p.life = 1 + rand() * 0.8;
      p.size = 0.15 + rand() * 0.15;
    },
  });
  sparks.emitting = false;
  const puff = smoke(scene, 1.15, 3.0, -1.5);
  puff.emitting = false;
  particles(scene, {
    texture: puffTexture(),
    count: 20,
    rate: 3,
    grow: 2,
    colors: ["#bbb4c8", "#e8e4f0"],
    spawn: (p, rand) => {
      p.sprite.position.set(1.15, 3.0, -1.5);
      p.velocity.set(0.1 + rand() * 0.1, 0.4 + rand() * 0.2, 0);
      p.life = 2.4 + rand();
      p.size = 0.2;
    },
  });
  let burst = false;
  scene.onUpdate((t) => {
    const c = loop(t, OVEN_LOOP);
    const p = push(c);
    const shut = doorClose(c);
    stove.door.rotation.y = -1.9 * (1 - shut);
    fire.forEach((f, i) => f.scale.set(1 + Math.sin(t * 13 + i * 2) * 0.12, 1 + Math.sin(t * 17 + i) * 0.22 + (1 - shut) * 0.1, 1));
    light.intensity = (1 - shut) * (14 + Math.sin(t * 21) * 3) + 3;
    witch.visible = witchShowing(c);
    setFace(witch, witchLooks, p > 0.05 ? "falling" : "wicked");
    witch.rotation.x = -witchLean(c) * 0.35 * (1 - p);
    witch.position.z = -0.95 * p;
    witch.position.y = 0.05 * p;
    witch.scale.setScalar(1 - p * 0.4);
    const run = mariaRun(c);
    maria.position.x = -2.6 * (1 - run);
    maria.position.z = -0.35 * smooth(c, 2.9, 3.4) * (1 - smooth(c, 5.0, 6.0));
    maria.position.y = run < 1 ? Math.abs(Math.sin(t * 12)) * 0.08 : 0;
    setFace(maria, mariaLooks, c > DOOR_SHUT ? "joy" : "run");
    setFace(joao, joaoLooks, c > DOOR_SHUT ? "joy" : "scared");
    joao.position.y = c > DOOR_SHUT ? Math.abs(Math.sin(t * 6)) * 0.08 : 0;
    if (c > DOOR_SHUT && !burst) {
      sparks.burst(60);
      burst = true;
    }
    if (c < DOOR_SHUT) burst = false;
    puff.emitting = c > DOOR_SHUT && c < DOOR_SHUT + 1.5;
  });
  return scene;
};

export const theEnd: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "meadow");
  sky(scene, "day");
  trees(scene, [[3.5, -1.6, 2.3], [-3.6, -0.3, 2.0]]);
  const banner = scene.cutout(paintBanner(["Fim"], "#e8505b"), 0.95, 0.4, -1.9, { delay: 0.05 });
  banner.position.y = 1.9;
  const home = cottage();
  scene.solid(home.group, -2.1, -1.2, 0.15, 0.25);
  scene.solid(fence(1.4), -2.0, 0.1, 0.4);
  smoke(scene, -2.55, 2.1, -1.3);
  const father = scene.cutout(paintFather("joy"), 1.7, -0.5, 0.1);
  const kids = [scene.cutout(child("joao", "joy", { wave: true }), 1.1, 0.5, 0.95), scene.cutout(child("maria", "joy", { wave: true }), 1.05, 1.3, 1.2)];
  const chest = treasureChest();
  chest.group.scale.setScalar(1.35);
  scene.solid(chest.group, 2.5, 0.7, 0.5, -0.45);
  scene.solid(mushroom(), -3.0, 1.7, 0.6);
  const coins = particles(scene, {
    texture: toTexture(paintCoin()),
    count: 40,
    rate: 0,
    gravity: 3,
    spawn: (p, rand) => {
      p.sprite.position.set(2.5 + (rand() - 0.5) * 0.4, 1.0, 0.7);
      p.velocity.set((rand() - 0.5) * 1.0, 1.6 + rand() * 0.9, (rand() - 0.2) * 0.6);
      p.life = 0.9 + rand() * 0.4;
      p.size = 0.11;
      p.spin = (rand() - 0.5) * 8;
    },
  });
  const shine = particles(scene, {
    texture: ctx.assets.spark,
    count: 40,
    rate: 16,
    additive: true,
    colors: ["#ffe27a", "#fff3b0", "#ffc93c"],
    spawn: (p, rand) => {
      p.sprite.position.set(2.5 + (rand() - 0.5) * 0.9, 1.0, 0.7 + (rand() - 0.5) * 0.5);
      p.velocity.set((rand() - 0.5) * 0.15, 0.35 + rand() * 0.3, 0);
      p.life = 0.8 + rand() * 0.6;
      p.size = 0.1 + rand() * 0.12;
    },
  });
  shine.emitting = false;
  particles(scene, {
    texture: toTexture(paintHeart()),
    count: 30,
    rate: 4,
    spawn: (p, rand) => {
      p.sprite.position.set(-1.2 + rand() * 2.4, 1.2, 0.6 + rand() * 0.4);
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
  let spilled = false;
  scene.onUpdate((t) => {
    if (t > next) {
      origin = { x: -2.8 + Math.random() * 5.6, y: 2.6 + Math.random() * 0.8 };
      fireworks.burst(50);
      next = t + 1.1;
    }
    const c = loop(t, 5);
    const open = smooth(c, 0.3, 1.0) * (1 - smooth(c, 4.2, 4.9));
    chest.lid.rotation.x = -1.95 * open;
    chest.glow.intensity = open * 4;
    shine.emitting = open > 0.6;
    if (open > 0.9 && !spilled) {
      coins.burst(12);
      spilled = true;
    }
    if (open < 0.1) spilled = false;
    banner.rotation.z = Math.sin(t * 1.4) * 0.03;
    father.rotation.z = Math.sin(t * 2) * 0.05;
    kids.forEach((k, i) => {
      k.position.y = Math.abs(Math.sin(t * 5 + i)) * 0.3;
      k.rotation.z = Math.sin(t * 5 + i) * 0.16;
    });
  });
  return scene;
};
