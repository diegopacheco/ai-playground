import { paintFather, paintWitch } from "../art/characters";
import { paintBanner, paintBird, paintBush, paintMoon, paintOwl } from "../art/props";
import { PAGE_Y } from "../engine/popup";
import { candyCane, candyHouse, cottage, crumb, fence, glowCap, logPile, lollipop, mushroom, rock, stump } from "./models";
import { type SceneBuilder, bob, butterflies, child, createScene, faces, fireflies, loop, pines, setFace, sky, smoke, sparkles, trees } from "./kit";
import { CRUMBS, CRUMB_LOOP, birdX, crumbShowing, joaoX } from "./timelines";

export const cover: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "cover");
  sky(scene, "night");
  trees(scene, [[-3.3, -1.6, 2.4], [3.4, -1.7, 2.3]]);
  const house = candyHouse();
  house.group.scale.setScalar(0.7);
  scene.solid(house.group, 2.3, -1.0, 0.15, -0.3);
  const witch = scene.cutout(paintWitch("wicked"), 1.7, -2.7, -1.2);
  const banner = scene.cutout(paintBanner(["João e Maria"], "#e8505b"), 1.25, 0, -0.7, { delay: 0.2 });
  banner.position.y = 1.15;
  const kids = [scene.cutout(child("joao", "joy", { wave: true }), 1.25, -0.8, 1.0), scene.cutout(child("maria", "joy", { wave: true }), 1.2, 0.8, 1.05)];
  scene.solid(lollipop("#ff5d8f", "#ffffff", 0.7), -3.1, 1.3, 0.6);
  scene.solid(lollipop("#7ad7ff", "#ffffff", 0.55), 3.2, 1.5, 0.65);
  scene.solid(candyCane(0.8), 2.4, 1.7, 0.6);
  scene.solid(mushroom(), -2.2, 1.9, 0.65);
  sparkles(scene, ctx.assets, [-3.8, 3.8, 0.4, 3.2], ["#ffe27a", "#ffd0f0", "#ffffff"]);
  scene.onUpdate((t) => {
    banner.rotation.z = Math.sin(t * 1.3) * 0.03;
    banner.position.y = 1.15 + Math.sin(t * 1.7) * 0.06;
    witch.position.x = 0.35 + Math.sin(t * 0.8) * 0.45;
    witch.rotation.z = Math.sin(t * 0.8) * 0.08;
    kids.forEach((k, i) => bob(k, 3.2, 0.25, i * 1.3)(t));
  });
  return scene;
};

export const woodcutter: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "meadow");
  sky(scene, "day");
  const home = cottage();
  scene.solid(home.group, -2.3, -1.1, 0.15, 0.25);
  scene.solid(fence(1.6), -2.2, 0.2, 0.4);
  smoke(scene, -2.75, 2.2, -1.2);
  trees(scene, [[2.9, -1.7, 2.3], [3.6, -0.5, 1.9]]);
  scene.solid(stump(), 0.7, 0.1, 0.45);
  scene.solid(logPile(), 2.5, 0.9, 0.5, 0.4);
  const father = scene.cutout(paintFather("sad"), 1.7, 1.5, -0.3);
  const kids = [scene.cutout(child("joao", "smile", { wave: true }), 1.15, -0.9, 1.0), scene.cutout(child("maria", "smile", { prop: "basket" }), 1.1, -0.1, 1.3)];
  scene.cutout(paintBush(2), 0.6, -3.5, 1.6);
  butterflies(scene, 3);
  scene.onUpdate((t) => {
    const swing = Math.pow(Math.abs(Math.sin(t * 2.2)), 4);
    father.rotation.z = -0.12 + swing * 0.2;
    father.position.y = swing * 0.05;
    kids.forEach((k, i) => bob(k, 4, 0.18, i * 1.7)(t));
  });
  return scene;
};

export const crumbs: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "forest");
  sky(scene, "day");
  pines(scene, [[-3.5, -1.8, 2.4], [-2.6, -1.5, 2.0], [-0.9, -1.9, 2.2], [1.2, -1.7, 2.3], [2.8, -1.5, 1.9], [3.6, -1.9, 2.4]]);
  scene.solid(rock(0.25, 3), -3.4, 2.1, 0.6);
  scene.solid(mushroom(), 3.4, 2.0, 0.6);
  const joao = scene.cutout(child("joao", "smile", { prop: "bread" }), 1.15, 0, 0.55);
  const maria = scene.cutout(child("maria", "smile"), 1.1, 0, 0.3);
  const birdFaces = faces({ up: paintBird("#5bb0f0", false), peck: paintBird("#5bb0f0", true) });
  const redFaces = faces({ up: paintBird("#f06a5b", false), peck: paintBird("#f06a5b", true) });
  const birds = [scene.cutout(paintBird("#5bb0f0", false), 0.5, 0, 0.85), scene.cutout(paintBird("#f06a5b", false), 0.45, 0, 1.05)];
  const trail = CRUMBS.map((x, i) => {
    const piece = crumb(i + 1);
    piece.scale.multiplyScalar(1.6);
    piece.position.set(x, PAGE_Y + 0.03, 0.75 + (i % 3) * 0.08);
    scene.free(piece);
    return piece;
  });
  scene.onUpdate((t) => {
    const c = loop(t, CRUMB_LOOP);
    const jx = joaoX(c);
    joao.position.x = jx;
    maria.position.x = jx + 0.75;
    bob(joao, 6, 0.08)(t);
    bob(maria, 6, 0.08, 1)(t);
    birds.forEach((bird, i) => {
      const bx = birdX(c) - i * 0.45;
      bird.position.x = bx;
      bird.position.y = Math.abs(Math.sin(t * 9 + i)) * 0.1;
      setFace(bird, i === 0 ? birdFaces : redFaces, Math.sin(t * 9 + i) > 0.4 ? "peck" : "up");
    });
    trail.forEach((piece, i) => {
      const x = CRUMBS[i] ?? 0;
      piece.visible = crumbShowing(c, x);
      const sinceDrop = (jx - x) / 0.35;
      piece.position.y = PAGE_Y + 0.03 + Math.max(0, 1 - sinceDrop) * 0.55;
    });
  });
  return scene;
};

export const lost: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "night");
  sky(scene, "night");
  const moon = scene.cutout(paintMoon(), 0.9, -2.7, -2.3, { delay: 0.05 });
  moon.position.y = 1.8;
  pines(scene, [[-3.5, -1.6, 2.5], [-1.6, -1.9, 2.2], [0.5, -1.8, 2.4], [2.0, -1.3, 2.0], [3.6, -1.7, 2.6], [-3.0, 0.4, 1.8]]);
  const owlFaces = faces({ open: paintOwl(true), shut: paintOwl(false) });
  const owl = scene.cutout(paintOwl(true), 0.75, 2.6, -0.6, { delay: 0.5 });
  owl.position.y = 1.3;
  const kids = [scene.cutout(child("maria", "scared"), 1.1, -0.35, 0.8), scene.cutout(child("joao", "sad"), 1.15, 0.35, 0.95)];
  const colors = ["#7affd8", "#b38cff", "#7ad7ff", "#ff9fd0"];
  [[-2.2, 1.2], [-1.6, 1.8], [1.6, 1.5], [2.4, 1.9], [1.1, 2.1], [-2.8, 2.0]].forEach(([x, z], i) => scene.solid(glowCap(colors[i % colors.length] ?? "#7affd8"), x ?? 0, z ?? 0, 0.6));
  scene.solid(rock(0.3, 5), 3.3, 0.6, 0.55);
  scene.solid(rock(0.22, 9), -3.5, 1.5, 0.6);
  fireflies(scene, ctx.assets);
  scene.onUpdate((t) => {
    setFace(owl, owlFaces, loop(t, 3.2) > 2.95 ? "shut" : "open");
    owl.rotation.z = Math.sin(t * 0.7) * 0.15;
    kids.forEach((k, i) => {
      k.position.x = Math.sin(t * 35 + i) * 0.012;
      k.rotation.z = (i === 0 ? 0.06 : -0.06) + Math.sin(t * 1.4) * 0.02;
    });
  });
  return scene;
};
