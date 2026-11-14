import { Group } from "three";
import { paintMother, paintOldMan } from "../art/characters";
import { paintBanner, paintBubble, paintBush, paintCow, paintMoon } from "../art/props";
import { PAGE_Y } from "../engine/popup";
import { bean, beanstalk, castle, cloudPuff, cottage, fence, logPile, mushroom, rock, stump } from "./models";
import { type SceneBuilder, bob, butterflies, createScene, faces, fireflies, jack, loop, setFace, sky, smoke, sparkles, trees } from "./kit";
import { LEAF_SPOTS, growth, smooth } from "./timelines";

export const cover: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "cover");
  sky(scene, "night");
  const tower = new Group();
  const stalk = beanstalk(2.3, LEAF_SPOTS);
  stalk.grow(1, () => 1);
  const cloud = cloudPuff(4, 1.3);
  cloud.position.y = 2.3 - 0.1;
  const keep = castle(ctx.assets.bricks);
  keep.group.scale.setScalar(0.32);
  keep.group.position.set(0, 2.3 + 0.15, -0.1);
  tower.add(stalk.group, cloud, keep.group);
  scene.solid(tower, 2.6, -1.4, 0.15);
  trees(scene, [[-3.4, -1.7, 2.3]]);
  const banner = scene.cutout(paintBanner(["Jack and the", "Beanstalk"], "#3f9f4a"), 1.6, -0.6, -0.9, { delay: 0.2 });
  banner.position.y = 0.9;
  const hero = scene.cutout(jack("joy", { pose: "wave" }), 1.3, -1.0, 1.0);
  const cow = scene.cutout(paintCow(true), 0.9, 0.6, 1.2);
  scene.solid(mushroom(), -2.6, 1.8, 0.65);
  scene.solid(rock(0.25, 4), 3.4, 1.6, 0.6);
  sparkles(scene, ctx.assets, [-3.8, 3.8, 0.4, 3.4], ["#d8ff7a", "#fff27a", "#ffffff"]);
  scene.lookHeight = 1.3;
  scene.onUpdate((t) => {
    banner.rotation.z = Math.sin(t * 1.3) * 0.03;
    banner.position.y = 0.9 + Math.sin(t * 1.7) * 0.06;
    tower.rotation.y = Math.sin(t * 0.4) * 0.15;
    bob(hero, 3.2, 0.25)(t);
    cow.rotation.z = Math.sin(t * 1.1) * 0.04;
  });
  return scene;
};

export const cottageScene: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "meadow");
  sky(scene, "day");
  const home = cottage();
  scene.solid(home.group, -2.2, -1.1, 0.15, 0.25);
  smoke(scene, -2.65, 2.2, -1.2);
  scene.solid(fence(2.0), 1.6, -0.2, 0.4);
  trees(scene, [[2.9, -1.8, 2.3], [3.7, -0.7, 1.9]]);
  scene.solid(logPile(), -3.4, 0.6, 0.5, 0.4);
  scene.solid(stump(), 3.2, 1.6, 0.55);
  const mother = scene.cutout(paintMother("sad"), 1.6, -0.6, -0.1);
  const hero = scene.cutout(jack("sad"), 1.15, 0.3, 0.6);
  const cowFaces = faces({ sad: paintCow(false), happy: paintCow(true) });
  const cow = scene.cutout(paintCow(false), 0.95, 1.8, 1.2);
  scene.cutout(paintBush(2), 0.6, -3.5, 1.7);
  butterflies(scene, 2);
  scene.onUpdate((t) => {
    mother.rotation.z = Math.sin(t * 0.9) * 0.03;
    bob(hero, 2.4, 0.08)(t);
    cow.rotation.z = Math.sin(t * 0.8) * 0.05;
    setFace(cow, cowFaces, loop(t, 4) > 3.6 ? "happy" : "sad");
  });
  return scene;
};

export const beansScene: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "forest");
  sky(scene, "day");
  trees(scene, [[-3.5, -1.8, 2.4], [-1.9, -1.9, 2.0], [2.2, -1.9, 2.2], [3.6, -1.5, 2.4]], true);
  const elder = scene.cutout(paintOldMan(), 1.6, 1.3, 0.1);
  const hero = scene.cutout(jack("smile"), 1.15, -0.6, 0.6);
  const cow = scene.cutout(paintCow(true), 0.95, -2.2, 0.4);
  const says = scene.cutout(paintBubble(["Five magic", "beans!"], "right"), 0.95, 0.6, -0.6, { delay: 0.6 });
  says.position.y = 1.55;
  const glowing = Array.from({ length: 5 }, (_, i) => {
    const b = bean();
    b.scale.multiplyScalar(1.4);
    scene.free(b);
    return { b, i };
  });
  scene.solid(rock(0.25, 3), -3.4, 2.0, 0.6);
  scene.solid(mushroom(), 3.4, 2.0, 0.6);
  sparkles(scene, ctx.assets, [0.4, 1.4, 1.0, 1.8], ["#d8ff7a", "#b6ff7a", "#ffffff"]);
  scene.onUpdate((t) => {
    elder.rotation.z = Math.sin(t * 1.2) * 0.05;
    bob(hero, 3, 0.1)(t);
    cow.rotation.z = Math.sin(t * 0.7) * 0.04;
    says.scale.setScalar(1 + Math.sin(t * 2) * 0.03);
    for (const { b, i } of glowing) {
      const a = t * 1.6 + (i / 5) * Math.PI * 2;
      b.position.set(0.55 + Math.cos(a) * 0.3, PAGE_Y + 1.25 + Math.sin(t * 2 + i) * 0.08, 0.45 + Math.sin(a) * 0.2);
    }
  });
  return scene;
};

export const windowScene: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "night");
  sky(scene, "night");
  const moon = scene.cutout(paintMoon(), 0.9, 2.6, -2.3, { delay: 0.05 });
  moon.position.y = 1.8;
  const home = cottage();
  scene.solid(home.group, -1.6, -0.9, 0.15, 0.15);
  trees(scene, [[-3.6, -1.6, 2.4], [3.6, -1.2, 2.2]]);
  const mother = scene.cutout(paintMother("angry"), 1.5, -0.4, 0.5);
  const hero = scene.cutout(jack("sad"), 1.1, -2.6, 0.9);
  const thrown = Array.from({ length: 5 }, (_, i) => {
    const b = bean();
    scene.free(b);
    return { b, i };
  });
  fireflies(scene, ctx.assets);
  scene.onUpdate((t) => {
    const c = loop(t, 5);
    const arms = smooth(c, 0.2, 0.6) * (1 - smooth(c, 1.4, 2.0));
    mother.rotation.z = -0.08 + arms * 0.14;
    mother.position.y = arms * 0.1;
    hero.rotation.z = Math.sin(t * 1.4) * 0.04;
    for (const { b, i } of thrown) {
      const fly = smooth(c, 0.5 + i * 0.08, 1.8 + i * 0.08);
      const x = -0.2 + fly * (1.4 + i * 0.35);
      b.position.set(x, PAGE_Y + 0.05 + 1.3 * fly * (1 - fly) * 3 + (1 - fly) * 0.9, 0.6 + i * 0.18);
      b.scale.setScalar(1.6 + Math.sin(t * 5 + i) * 0.3 * fly);
    }
  });
  return scene;
};

export const beanstalkScene: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "meadow");
  sky(scene, "day");
  const home = cottage();
  home.group.scale.setScalar(0.75);
  scene.solid(home.group, -2.8, -1.3, 0.15, 0.3);
  trees(scene, [[-3.8, 0.2, 1.8]]);
  const heroFaces = faces({ scared: jack("scared"), joy: jack("joy", { pose: "wave" }) });
  const hero = scene.cutout(jack("scared"), 1.15, 0.6, 0.9);
  const mother = scene.cutout(paintMother("smile"), 1.4, -0.8, 0.6);
  const cow = scene.cutout(paintCow(false), 0.85, -2.2, 1.6);
  const whoosh = scene.cutout(paintBubble(["WHOOSH!"], "left"), 0.9, 2.6, 0.4, { delay: 0.6 });
  whoosh.position.y = 1.2;
  sparkles(scene, ctx.assets, [2.0, 4.6, 0.4, 4.0], ["#d8ff7a", "#b6ff7a", "#ffffff"]);
  scene.lookHeight = 1.2;
  scene.onUpdate((t) => {
    const g = growth(scene.since);
    whoosh.visible = g > 0.01 && g < 0.97;
    whoosh.rotation.z = Math.sin(t * 6) * 0.06;
    setFace(hero, heroFaces, g > 0.97 ? "joy" : "scared");
    bob(hero, 4, g > 0.97 ? 0.25 : 0.05)(t);
    mother.rotation.z = Math.sin(t * 0.8) * 0.04;
    cow.rotation.z = Math.sin(t * 1.6) * 0.05;
  });
  return scene;
};
