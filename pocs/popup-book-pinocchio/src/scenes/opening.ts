import { BoxGeometry, Mesh, MeshStandardMaterial, Vector3 } from "three";
import { FLOOR_Y } from "../engine/popup";
import { toTexture } from "../art/canvas";
import { paintBanner, paintBook, paintBubble, paintCloud } from "../art/props";
import { paintCricket, paintGeppetto, paintPinocchio } from "../art/puppets";
import { log, workbench } from "./models";
import { type SceneBuilder, actor, chips, createScene, dangle, every, notes, pinocchio, pop, smooth, sparkles, stars, strings } from "./kit";

export const cover: SceneBuilder = (ctx) => {
  const scene = createScene("night", { key: "#ffe2b0", fill: "#b7a0ff", ambient: 0.45 });
  const banner = actor(scene, { canvas: paintBanner(["Pinocchio"], "#c42f2f"), at: { l: [0.12, 0.1], r: [0.88, 0.1] } }, 1.1, 0, -0.9, { delay: 0.2 });
  strings(scene, banner, ["l", "r"]);
  const clouds = [-2.8, 2.9].map((x, i) => {
    const cloud = actor(scene, { canvas: paintCloud("#e8e4ff"), at: { c: [0.5, 0.1] } }, 0.7, x, -1.6, { delay: 0.1 });
    strings(scene, cloud, ["c"]);
    return { cloud, i };
  });
  const hero = pinocchio(scene, { mood: "joy", pose: "dance" }, 1.9, 0, 0.7);
  strings(scene, hero, ["head", "handL", "handR"]);
  const cricket = actor(scene, paintCricket(), 0.7, 2.4, 1.2);
  sparkles(scene, ctx.assets, [-4, 4, 0.5, 4.5], ["#ffe27a", "#ffd0f0", "#ffffff"]);
  scene.onUpdate((t) => {
    banner.mesh.position.y = 2.5 + Math.sin(t * 1.4) * 0.08;
    banner.mesh.rotation.z = Math.sin(t * 1.1) * 0.04;
    for (const { cloud, i } of clouds) {
      cloud.mesh.position.y = 3.2 + Math.sin(t * 0.8 + i * 2) * 0.12;
    }
    dangle(hero.mesh, t, 3.4, 0.45);
    hero.mesh.position.x = Math.sin(t * 0.7) * 1.6;
    hero.mesh.scale.x = Math.cos(t * 0.7) >= 0 ? 1 : -1;
    dangle(cricket.mesh, t, 5, 0.12, 1);
    scene.focus.set(hero.mesh.position.x, 1.4, 0.7);
  });
  return scene;
};

export const workshop: SceneBuilder = (ctx) => {
  const scene = createScene("workshop", { key: "#ffe2b0", fill: "#ffb870", ambient: 0.55 });
  const bench = scene.solid(workbench(), 0.8, -0.5, 0.15);
  const wood = log(1.5);
  wood.position.y = 0.98;
  bench.add(wood);
  const geppetto = actor(scene, paintGeppetto("smile", "mallet"), 2.3, -1.4, 0.3);
  strings(scene, geppetto, ["head", "hand"]);
  const ouch = scene.cutout(paintBubble("Ouch!"), 0.7, 2.4, -0.2, { delay: 0.5 });
  const flying = chips(scene, 0.6, 1.3, -0.3);
  const strike = every(0.9, 0.3);
  const shout = every(2.7, 0.3);
  let shoutAt = -10;
  sparkles(scene, ctx.assets, [-0.4, 1.8, 1.0, 2.2], ["#ffe9a8"], 3);
  scene.onUpdate((t, dt) => {
    const phase = (t % 0.9) / 0.9;
    geppetto.mesh.rotation.z = phase < 0.3 ? -0.12 * (phase / 0.3) : -0.12 + 0.2 * smooth(phase, 0.3, 0.42) - 0.08 * smooth(phase, 0.5, 1);
    if (strike(t, dt)) flying.burst(10);
    if (shout(t, dt)) shoutAt = t;
    const since = t - shoutAt;
    pop(ouch, since < 1.3 ? Math.min(since * 6, 1) * (1 - smooth(since, 1.0, 1.3)) : 0);
    ouch.position.y = 1.4;
    wood.rotation.z = Math.sin(t * 40) * 0.01 * (since < 0.4 ? 1 : 0);
    scene.focus.set(0, 1.4, 0);
  });
  return scene;
};

export const alive: SceneBuilder = (ctx) => {
  const scene = createScene("sunroom", { key: "#fff1c8", fill: "#ffcf7a", ambient: 0.6 });
  scene.solid(workbench(), 2.3, -1.2, 0.15, -0.3);
  const geppetto = actor(scene, paintGeppetto("scared"), 2.2, -2.7, -0.2);
  strings(scene, geppetto, ["head", "hand"]);
  const hero = pinocchio(scene, { mood: "joy", pose: "dance" }, 1.7, 0, 0.6);
  strings(scene, hero, ["head", "handL", "handR"]);
  const poses = [paintPinocchio({ mood: "joy", pose: "dance" }), paintPinocchio({ mood: "joy", pose: "stand" })].map((f) => toTexture(f.canvas));
  const burst = stars(scene, "#ffd23f", () => new Vector3(hero.mesh.position.x, FLOOR_Y + 1.2 + hero.mesh.position.y, 0.6 + hero.mesh.position.z));
  const hop = every(1.6);
  notes(scene, 0, 1.8, 0.6, ["#ffffff", "#ffe27a", "#ff9fd0"]);
  sparkles(scene, ctx.assets, [-3.5, 3.5, 0.3, 3.8], ["#ffffff", "#ffe27a"], 8);
  scene.onUpdate((t, dt) => {
    const a = t * 0.9;
    hero.mesh.position.x = Math.sin(a) * 2.3;
    hero.mesh.position.z = Math.cos(a) * 0.5;
    hero.mesh.scale.x = Math.cos(a) >= 0 ? 1 : -1;
    dangle(hero.mesh, t, 4.2, 0.5);
    const map = poses[Math.floor(t * 2) % 2];
    if (map && hero.mesh.material.map !== map) {
      hero.mesh.material.map = map;
      hero.mesh.material.needsUpdate = true;
    }
    if (hop(t, dt)) burst.burst(14);
    geppetto.mesh.position.y = Math.abs(Math.sin(t * 9)) * 0.04;
    geppetto.mesh.rotation.z = Math.sin(t * 1.2) * 0.05;
    scene.focus.set(hero.mesh.position.x, 1.3, 0.6);
  });
  return scene;
};

export const cricketScene: SceneBuilder = () => {
  const scene = createScene("kitchen", { key: "#ffd9a0", fill: "#ff9a4a", ambient: 0.45 });
  const stool = new Mesh(new BoxGeometry(0.9, 0.9, 0.7), new MeshStandardMaterial({ color: "#7a4d22", roughness: 0.8 }));
  stool.position.y = 0.45;
  scene.solid(stool, 2.3, -0.4, 0.2);
  const cricket = actor(scene, paintCricket(), 1.0, 2.3, -0.2, { delay: 0.5 });
  strings(scene, cricket, ["head"]);
  const bubble = scene.cutout(paintBubble("Go to school!"), 0.85, 1.6, -0.6, { delay: 0.55 });
  const hero = pinocchio(scene, { mood: "sad", pose: "ears" }, 1.8, -1.6, 0.6);
  strings(scene, hero, ["head", "handL", "handR"]);
  scene.cutout(paintBook(), 0.45, -0.3, 1.3, { turn: 0.3 });
  notes(scene, 2.4, 1.9, -0.2, ["#9dff9a", "#ffffff", "#d8ff7a"]);
  scene.onUpdate((t) => {
    cricket.mesh.position.y = 0.9 + Math.abs(Math.sin(t * 6)) * 0.08;
    cricket.mesh.scale.setScalar(1 + Math.max(Math.sin(t * 12), 0) * 0.04);
    const talk = (t % 4) / 4;
    pop(bubble, smooth(talk, 0.05, 0.15) * (1 - smooth(talk, 0.7, 0.8)));
    bubble.position.y = 2.1;
    hero.mesh.rotation.z = Math.sin(t * 7) * 0.12;
    hero.mesh.position.x = Math.sin(t * 0.8) * 0.4;
    scene.focus.set(0.2, 1.3, 0.2);
  });
  return scene;
};
