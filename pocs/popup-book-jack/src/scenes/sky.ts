import { Vector3 } from "three";
import { paintGiantWife, paintMother } from "../art/characters";
import { toTexture } from "../art/canvas";
import { paintBanner, paintBubble, paintCoin, paintHarp, paintHeart, paintNote, paintPuff } from "../art/props";
import { PAGE_Y, type PopupScene, cutoutMesh } from "../engine/popup";
import { type Giant, castle, cloudPuff, giant3d, goldenEgg, giantTable, hen3d, oven, rainbow, rock, stalkPoint, treasureChest } from "./models";
import { type SceneBuilder, bob, createScene, jack, loop, particles, sparkles, trees } from "./kit";
import { STALK_BASE, STALK_GIRTH, STATIONS } from "./stations";
import {
  CHASE_LOOP,
  EGG_LOOP,
  FALL_START,
  FALL_TIME,
  LAY_TIMES,
  STALK_HEIGHT,
  axeSwing,
  chopsDone,
  crashed,
  eggDrop,
  giantHeight,
  henSquat,
  jackHeight,
  smooth,
  stalkFall,
} from "./timelines";

function notes(scene: PopupScene, x: number, y: number, z: number): void {
  particles(scene, {
    texture: toTexture(paintNote()),
    count: 16,
    rate: 2.5,
    spawn: (p, rand) => {
      p.sprite.position.set(x + (rand() - 0.5) * 0.2, y, z);
      p.velocity.set((rand() - 0.3) * 0.4, 0.4 + rand() * 0.2, 0);
      p.life = 2 + rand();
      p.size = 0.16 + rand() * 0.06;
      p.spin = (rand() - 0.5) * 1.5;
    },
  });
}

function dustPuffs(scene: PopupScene, x: number, z: number, spread: number) {
  return particles(scene, {
    texture: toTexture(paintPuff()),
    count: 40,
    rate: 0,
    grow: 1.5,
    colors: ["#e8e0d0", "#d8cfc0"],
    spawn: (p, rand) => {
      p.sprite.position.set(x + (rand() - 0.5) * spread, PAGE_Y + 0.1, z + (rand() - 0.5) * 0.4);
      p.velocity.set((rand() - 0.5) * 1.2, 0.2 + rand() * 0.4, 0.3);
      p.life = 1 + rand() * 0.5;
      p.size = 0.28;
    },
  });
}

function walk(giant: Giant, t: number, speed: number, stride: number): number {
  const swing = Math.sin(t * speed);
  giant.legs[0].rotation.x = swing * stride;
  giant.legs[1].rotation.x = -swing * stride;
  giant.body.position.y = 1.35 + Math.abs(Math.cos(t * speed)) * 0.06;
  return swing;
}

function sitAsleep(giant: Giant): void {
  giant.group.position.y = -1.0;
  for (const leg of giant.legs) leg.rotation.x = -1.45;
  giant.legs[0].rotation.z = 0.15;
  giant.legs[1].rotation.z = -0.15;
  giant.body.rotation.x = -0.12;
  giant.head.rotation.x = 0.35;
  giant.head.rotation.z = 0.15;
  giant.arms[0].rotation.set(-0.6, 0, -0.3);
  giant.arms[1].rotation.set(-0.6, 0, 0.3);
  for (const lid of giant.lids) lid.visible = true;
}

export const cloudsScene: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "sky");
  const keep = castle(ctx.assets.bricks);
  scene.solid(keep.group, 0.4, -1.3, 0.2, -0.1);
  const arc = rainbow(3.4);
  scene.solid(arc, 0.4, -2.5, 0.1);
  const puffs = [[-3.4, -1.6, 1.4], [-0.6, -1.9, 1.1], [-3.0, 0.4, 1.0], [-1.0, 1.9, 1.2], [1.4, 1.6, 0.9], [3.4, 2.0, 0.9]] as const;
  const clouds = puffs.map(([x, z, s], i) => scene.solid(cloudPuff(i * 7 + 1, s), x, z, 0.3));
  const hero = scene.cutout(jack("smile"), 1.0, 3.2, 0.2);
  sparkles(scene, ctx.assets, [-3.8, 3.8, 0.8, 3.6], ["#ffffff", "#e8f6ff", "#fff27a"]);
  scene.lookHeight = 1.5;
  scene.onUpdate((t) => {
    const c = loop(t, 8);
    const go = smooth(c, 0.5, 4.0) - smooth(c, 5.0, 7.8);
    hero.position.x = -go * 2.2;
    hero.scale.x = c < 4.5 ? -1 : 1;
    bob(hero, 7, 0.08)(t);
    clouds.forEach((cloud, i) => {
      cloud.position.x = Math.sin(t * 0.3 + i) * 0.15;
      cloud.position.y = Math.sin(t * 0.5 + i * 2) * 0.05;
    });
    arc.rotation.y = Math.sin(t * 0.3) * 0.05;
    keep.door.rotation.y = -0.9 * smooth(c, 3.4, 4.0) * (1 - smooth(c, 7.0, 7.8));
  });
  return scene;
};

export const castleScene: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "castle");
  const stove = oven(ctx.assets.bricks);
  scene.solid(stove.group, -2.4, -1.3, 0.15, 0.25);
  const giant = giant3d();
  scene.solid(giant.group, 2.0, -1.0, 0.25, -0.35);
  const wife = scene.cutout(paintGiantWife(), 2.0, -0.1, -0.5);
  const hero = scene.cutout(jack("scared"), 0.75, -1.7, 0.3);
  const roar = scene.cutout(paintBubble(["Fee-fi-fo-fum!", "I smell a boy!"], "right"), 1.1, 0.6, 0.6, { delay: 0.6 });
  roar.position.y = 2.4;
  const dust = dustPuffs(scene, 2.0, -0.6, 1.2);
  let landed = 0;
  scene.lookHeight = 1.6;
  scene.onUpdate((t) => {
    const swing = walk(giant, t, 2.2, 0.4);
    const side = Math.sign(swing);
    if (side !== landed && Math.abs(swing) > 0.95) {
      dust.burst(10);
      landed = side;
    }
    scene.shake = Math.max(0, Math.abs(swing) - 0.85) * 3;
    const roaring = Math.max(0, Math.sin(t * 0.9));
    giant.arms[0].rotation.set(-0.3 - roaring * 0.6, 0, -0.4 - roaring * 1.9);
    giant.arms[1].rotation.set(-0.3 - roaring * 0.6, 0, 0.4 + roaring * 1.9);
    giant.jaw.scale.y = 0.3 + roaring * 1.2;
    giant.head.rotation.y = Math.sin(t * 0.7) * 0.4;
    giant.group.position.x = Math.sin(t * 0.45) * 0.5;
    wife.position.x = Math.sin(t * 40) * scene.shake * 0.02;
    roar.scale.setScalar(0.85 + roaring * 0.25);
    stove.door.rotation.y = -0.35 - Math.max(0, Math.sin(t * 0.9 + 2)) * 0.5;
    hero.position.x = -0.1 + Math.max(0, Math.sin(t * 0.9 + 2)) * 0.25;
  });
  return scene;
};

export const treasureScene: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "castle");
  const table = giantTable();
  const hen = hen3d();
  hen.group.position.set(-0.5, 0.98, 0);
  hen.group.rotation.y = -0.4;
  table.add(hen.group);
  scene.solid(table, -0.5, -0.9, 0.2);
  const harp = scene.cutout(paintHarp(), 0.6, 0.3, -0.85, { delay: 0.5 });
  harp.position.y = 0.96;
  notes(scene, 0.3, PAGE_Y + 1.6, -0.8);
  const giant = giant3d();
  sitAsleep(giant);
  scene.solid(giant.group, 2.6, -1.0, 0.3, -0.5);
  const snore = scene.cutout(paintBubble(["Zzz..."], "right"), 0.6, 2.0, -0.4, { delay: 0.6 });
  snore.position.y = 2.3;
  const hero = scene.cutout(jack("joy"), 1.0, -2.8, 0.7);
  const eggs = LAY_TIMES.map(() => {
    const egg = goldenEgg();
    scene.free(egg);
    return egg;
  });
  sparkles(scene, ctx.assets, [-1.4, -0.6, 1.2, 1.8], ["#ffe27a", "#fff3b0"]);
  scene.onUpdate((t) => {
    const c = loop(t, EGG_LOOP);
    const squat = henSquat(c);
    hen.body.position.y = 0.24 - squat * 0.1;
    hen.body.scale.set(1 + squat * 0.12, 1 - squat * 0.12, 1 + squat * 0.12);
    hen.head.rotation.z = Math.sin(t * 6) * 0.25 * (1 - squat);
    eggs.forEach((egg, i) => {
      const drop = eggDrop(c, i);
      egg.visible = drop > 0 && scene.openness > 0.9;
      egg.position.set(-1.3 + i * 0.17, PAGE_Y + 1.02 + (1 - drop) * 0.12, -0.75 + (i % 2) * 0.1);
      egg.rotation.y = t;
    });
    harp.rotation.z = Math.sin(t * 2) * 0.03;
    const breath = Math.sin(t * 1.1);
    giant.body.scale.set(1 + breath * 0.03, 1 + breath * 0.02, 1 + breath * 0.04);
    giant.head.rotation.x = 0.35 + breath * 0.05;
    snore.scale.setScalar(0.8 + Math.max(breath, 0) * 0.35);
    bob(hero, 2, 0.06)(t);
  });
  return scene;
};

function onStalk(height: number, out: number): Vector3 {
  const station = STATIONS.chase.at;
  const point = stalkPoint(height / STALK_HEIGHT, STALK_HEIGHT, STALK_GIRTH);
  return new Vector3(STALK_BASE.x + point.x - station.x, STALK_BASE.y + point.y - station.y, STALK_BASE.z + point.z - station.z + out);
}

export const chaseScene: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "sky");
  [[-2.8, -1.4, 1.2], [2.9, -1.0, 1.1], [-2.6, 1.6, 0.9], [2.6, 1.8, 1.0]].forEach(([x, z, s], i) => scene.solid(cloudPuff(i * 5 + 2, s ?? 1), x ?? 0, z ?? 0, 0.3));
  const hero = cutoutMesh(jack("scared", { pose: "climb" }), 1.0);
  const harp = cutoutMesh(paintHarp(), 0.5);
  const giant = giant3d();
  giant.group.scale.setScalar(0.8);
  for (const piece of [hero, harp, giant.group]) scene.free(piece);
  const cry = scene.cutout(paintBubble(["Master!", "Master!"], "left"), 0.8, -2.0, 0.4, { delay: 0.6 });
  cry.position.y = 2.6;
  const stomp = scene.cutout(paintBubble(["STOMP!"], "right"), 0.75, 2.0, 0.4, { delay: 0.6 });
  stomp.position.y = 2.4;
  const station = STATIONS.chase.at;
  scene.lookHeight = 5.0;
  scene.onUpdate((t) => {
    const c = loop(t, CHASE_LOOP);
    const show = scene.openness > 0.6;
    for (const piece of [hero, harp, giant.group]) piece.visible = show;
    const jackWorld = station.y + 0.6 + jackHeight(c) * 2.4;
    hero.position.copy(onStalk(jackWorld, 0.75));
    hero.rotation.z = Math.sin(t * 8) * 0.06;
    harp.position.copy(onStalk(jackWorld + 0.4, 1.0)).add(new Vector3(0.5, 0, 0));
    const giantWorld = station.y + 0.6 + giantHeight(c) * 2.4;
    giant.group.position.copy(onStalk(giantWorld, 0.9));
    const climb = Math.sin(t * 5);
    giant.arms[0].rotation.set(-2.4 + climb * 0.5, 0, -0.3);
    giant.arms[1].rotation.set(-2.4 - climb * 0.5, 0, 0.3);
    giant.legs[0].rotation.x = climb * 0.5;
    giant.legs[1].rotation.x = -climb * 0.5;
    giant.jaw.scale.y = 0.6 + Math.abs(climb) * 0.8;
    cry.rotation.z = Math.sin(t * 7) * 0.05;
    stomp.scale.setScalar(1 + Math.abs(Math.sin(t * 5)) * 0.1);
  });
  return scene;
};

export const theEnd: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "meadow");
  trees(scene, [[3.6, -1.7, 2.1]]);
  const hero = scene.cutout(jack("joy", { prop: "axe" }), 1.2, -3.0, -2.0, { flip: true });
  const mother = scene.cutout(paintMother("joy"), 1.5, 0.4, 0.2);
  const chest = treasureChest();
  scene.solid(chest.group, 2.4, 0.9, 0.5, -0.4);
  scene.solid(rock(0.22, 6), -3.4, 1.8, 0.6);
  const hen = hen3d();
  scene.solid(hen.group, -0.8, 1.3, 0.6, 0.6);
  const harp = scene.cutout(paintHarp(), 0.55, 1.4, 1.5);
  const chop = scene.cutout(paintBubble(["CHOP!"], "right"), 0.7, -1.6, -1.2, { delay: 0.6 });
  chop.position.y = 1.5;
  const crash = scene.cutout(paintBubble(["CRASH!"], "left"), 1.0, -1.2, -0.8, { delay: 0.6 });
  crash.position.y = 1.7;
  const banner = scene.cutout(paintBanner(["The End"], "#3f9f4a"), 0.9, 1.4, -1.6, { delay: 0.3 });
  banner.position.y = 1.9;
  const coins = particles(scene, {
    texture: toTexture(paintCoin()),
    count: 40,
    rate: 0,
    gravity: 3,
    spawn: (p, rand) => {
      p.sprite.position.set(2.4 + (rand() - 0.5) * 0.4, 1.0, 1.1);
      p.velocity.set((rand() - 0.5) * 1.0, 1.6 + rand() * 0.9, (rand() - 0.2) * 0.6);
      p.life = 0.9 + rand() * 0.4;
      p.size = 0.11;
      p.spin = (rand() - 0.5) * 8;
    },
  });
  particles(scene, {
    texture: toTexture(paintHeart()),
    count: 24,
    rate: 3,
    spawn: (p, rand) => {
      p.sprite.position.set(-0.4 + rand() * 1.6, 1.4, 0.4 + rand() * 0.4);
      p.velocity.set((rand() - 0.5) * 0.3, 0.6 + rand() * 0.4, 0);
      p.life = 2.5 + rand();
      p.size = 0.15 + rand() * 0.1;
    },
  });
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
      p.sprite.position.set(-2 + Math.floor(rand() * 3) * 2, 3.2, -1.8);
      p.velocity.set(Math.cos(a) * speed, Math.sin(a) * speed, 0);
      p.life = 1.3 + rand() * 0.4;
      p.size = 0.22;
    },
  });
  let spilled = 0;
  let boom = 0;
  scene.lookHeight = 1.6;
  scene.onUpdate((t) => {
    const since = scene.since;
    hero.rotation.z = axeSwing(since) * 0.35;
    chop.visible = chopsDone(since) > 0 && stalkFall(since) === 0;
    chop.scale.setScalar(0.8 + axeSwing(since) * 0.4);
    const down = crashed(since);
    crash.visible = down && since < FALL_START + FALL_TIME + 2.5;
    banner.visible = down;
    const open = smooth(since, FALL_START + FALL_TIME + 0.6, FALL_START + FALL_TIME + 1.3);
    chest.lid.rotation.x = -1.95 * open;
    chest.glow.intensity = open * 4;
    if (open > 0.9 && t > spilled) {
      coins.burst(10);
      spilled = t + 1.6;
    }
    if (open > 0.9 && t > boom) {
      fireworks.burst(45);
      boom = t + 1.2;
    }
    mother.rotation.z = Math.sin(t * 2) * 0.05;
    bob(mother, 4, open * 0.2)(t);
    hen.body.rotation.z = -Math.max(0, Math.sin(t * 3)) * 0.5;
    harp.rotation.z = Math.sin(t * 2) * 0.04;
    banner.rotation.z = Math.sin(t * 1.4) * 0.03;
  });
  return scene;
};
