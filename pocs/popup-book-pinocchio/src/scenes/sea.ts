import { PointLight, Vector3 } from "three";
import { toTexture } from "../art/canvas";
import { paintBanner, paintBubble, paintWave } from "../art/props";
import { paintCricket, paintFairy, paintGeppetto, paintPinocchio } from "../art/puppets";
import { FLOOR_Y, type PopupScene } from "../engine/popup";
import { balloon, candle, carousel, drum, ribs, seaweed, shark } from "./models";
import { type SceneBuilder, actor, bubbles, confetti, createScene, dangle, every, hearts, loop, pinocchio, pop, smooth, sparkles, stars, strings } from "./kit";
import { EARS_LOOP, SHARK_LOOP, earGrowth, heeHaw, jawOpen, swallowed, swimX } from "./timelines";

function swapMap(mesh: { material: { map: unknown; needsUpdate: boolean } }, map: unknown): void {
  if (mesh.material.map !== map) {
    mesh.material.map = map;
    mesh.material.needsUpdate = true;
  }
}

function waves(scene: PopupScene, colors: string[], height: number): void {
  const rows = colors.map((color, i) => ({ wave: scene.cutout(paintWave(color), height, 0, -1.6 + i * 1.5, { delay: 0.1 + i * 0.1 }), i }));
  scene.onUpdate((t) => {
    for (const { wave, i } of rows) {
      wave.position.x = Math.sin(t * (0.8 + i * 0.2) + i) * 0.35;
      wave.position.y = -0.05 + Math.sin(t * 1.6 + i * 2) * 0.05;
    }
  });
}

export const toyland: SceneBuilder = (ctx) => {
  const scene = createScene("carnival", { key: "#fff0f8", fill: "#ffb0e0", ambient: 0.6 });
  const ride = carousel();
  scene.solid(ride.group, 1.9, -0.9, 0.15);
  scene.solid(drum(), -3.3, 1.0, 0.5);
  scene.solid(drum(), 3.4, 1.2, 0.55);
  const floating = ["#e8505b", "#3fa8e8", "#ffd23f", "#4caf50", "#ff6fa3"].map((color, i) => {
    const b = balloon(color);
    scene.free(b);
    return { b, i };
  });
  const hero = pinocchio(scene, { mood: "joy", pose: "dance" }, 1.8, -1.2, 0.8, { ears: true });
  strings(scene, hero, ["head", "handL", "handR"]);
  const happy = toTexture(paintPinocchio({ mood: "joy", pose: "dance" }).canvas);
  const shocked = toTexture(paintPinocchio({ mood: "scared", pose: "ears" }).canvas);
  const bray = scene.cutout(paintBubble("HEE-HAW!"), 0.85, 0.2, 0.4, { delay: 0.6 });
  confetti(scene, 6);
  sparkles(scene, ctx.assets, [-4, 4, 2, 4.5], ["#ffffff", "#ffe27a"], 5);
  scene.onUpdate((t) => {
    const c = loop(t, EARS_LOOP);
    ride.spin.rotation.y = t * 0.8;
    ride.horses.forEach((horse, i) => {
      horse.position.y = 1.0 + Math.sin(t * 3 + i * 1.3) * 0.25;
    });
    for (const { b, i } of floating) {
      const rise = loop(t * 0.35 + i * 0.37, 1);
      b.position.set(-3 + i * 1.5 + Math.sin(t + i) * 0.3, FLOOR_Y + rise * 4.2, -1.2 + (i % 2) * 0.6);
      b.visible = scene.openness > 0.9;
    }
    const ears = earGrowth(c);
    hero.setEars(ears);
    swapMap(hero.mesh, ears > 0.5 ? shocked : happy);
    if (ears < 0.5) dangle(hero.mesh, t, 4, 0.35);
    else {
      hero.mesh.position.y = 0;
      hero.mesh.rotation.z = Math.sin(t * 18) * 0.04;
    }
    pop(bray, heeHaw(c) ? Math.min((c - 3.1) * 6, 1) : 0);
    bray.position.y = 2.4;
    scene.focus.set(0, 1.5, 0.3);
  });
  return scene;
};

export const sharkScene: SceneBuilder = () => {
  const scene = createScene("storm", { key: "#cfe0ff", fill: "#7aa0ff", ambient: 0.75 });
  const beast = shark();
  beast.group.scale.setScalar(0.8);
  scene.solid(beast.group, 2.4, -0.3, 0.1);
  const hero = pinocchio(scene, { mood: "scared", pose: "swim" }, 1.2, 0, 0.55);
  waves(scene, ["#1e5a8a", "#2a6a9a", "#3a7aaa"], 0.75);
  const gulp = scene.cutout(paintBubble("GULP!"), 0.8, 2.0, -0.6, { delay: 0.6 });
  const weeds = [[-3.6, 1.6, "#2f8a5a"], [3.7, 1.4, "#3a9a4a"], [-2.8, 1.9, "#4aa86a"]] as const;
  const swaying = weeds.map(([x, z, color]) => {
    const weed = seaweed(1.6, color);
    scene.solid(weed.group, x, z, 0.4);
    return weed.joints;
  });
  const flash = new PointLight("#e8f0ff", 0, 30, 1);
  flash.position.set(-1, 7, 2);
  scene.free(flash);
  bubbles(scene, [-4, 4], 6);
  const strike = every(5.3, 1.1);
  let lit = -10;
  scene.onUpdate((t, dt) => {
    const c = loop(t, SHARK_LOOP);
    const open = jawOpen(c);
    beast.upper.rotation.z = -open * 0.45;
    beast.lower.rotation.z = open * 0.4;
    beast.group.position.y = Math.sin(t * 1.3) * 0.08;
    beast.group.rotation.z = Math.sin(t * 0.9) * 0.03;
    hero.mesh.position.x = swimX(c);
    hero.mesh.position.y = 0.45 + Math.sin(t * 5) * 0.08;
    hero.mesh.rotation.z = Math.sin(t * 6) * 0.12;
    hero.mesh.visible = !swallowed(c);
    const since = c - 4.85;
    pop(gulp, since > 0 && since < 2.4 ? Math.min(since * 6, 1) * (1 - smooth(since, 2.0, 2.4)) : 0);
    gulp.position.y = 2.4;
    swaying.forEach((joints, i) => {
      joints.forEach((joint, k) => {
        joint.rotation.z = Math.sin(t * 1.5 + i * 2 - k * 0.5) * 0.18;
      });
    });
    if (strike(t, dt)) lit = t;
    const age = t - lit;
    flash.intensity = age < 0.35 ? (age < 0.08 || (age > 0.16 && age < 0.24) ? 400 : 0) : 0;
    scene.focus.set(swallowed(c) ? 1.6 : swimX(c) * 0.6 + 0.6, 1.2, 0);
  });
  return scene;
};

export const inside: SceneBuilder = () => {
  const scene = createScene("belly", { key: "#ffb0a0", fill: "#ff7a5a", ambient: 0.25 });
  scene.solid(ribs(5, 3.4), 0, 0, 0.05);
  const light = candle();
  scene.solid(light.group, 1.0, -0.2, 0.3);
  const father = actor(scene, paintGeppetto("sad", "none"), 2.1, 2.2, -0.5);
  strings(scene, father, ["head", "hand"]);
  const happy = toTexture(paintGeppetto("joy", "none").canvas);
  const sad = toTexture(paintGeppetto("sad", "none").canvas);
  const hero = pinocchio(scene, { mood: "joy", pose: "carry" }, 1.6, 0, 0.6);
  strings(scene, hero, ["head", "handL", "handR"]);
  waves(scene, ["#3a5a7a"], 0.5);
  const love = hearts(scene, 1.4, 2.4, 0.2);
  bubbles(scene, [-3, 3], 2);
  scene.onUpdate((t) => {
    const c = loop(t, 9);
    light.flame.scale.set(1 + Math.sin(t * 23) * 0.15, 1 + Math.sin(t * 17) * 0.25, 1);
    light.light.intensity = 6 + Math.sin(t * 19) * 1.2 + Math.sin(t * 7) * 0.8;
    const walk = smooth(c, 0.5, 3.6);
    hero.mesh.position.x = -3.2 + walk * 3.6;
    hero.mesh.position.y = walk < 1 ? Math.abs(Math.sin(t * 7)) * 0.1 : Math.abs(Math.sin(t * 4)) * 0.2;
    const together = c > 3.6 && c < 8;
    love.emitting = together;
    swapMap(father.mesh, together ? happy : sad);
    father.mesh.position.y = together ? Math.abs(Math.sin(t * 5)) * 0.15 : 0;
    father.mesh.rotation.z = together ? Math.sin(t * 2.5) * 0.08 : -0.05;
    scene.focus.set(hero.mesh.position.x * 0.5 + 0.8, 1.4, 0);
  });
  return scene;
};

export const theEnd: SceneBuilder = (ctx) => {
  const scene = createScene("sunrise", { key: "#fff3d0", fill: "#ffd0e0", ambient: 0.65 });
  const banner = actor(scene, { canvas: paintBanner(["The End"], "#3a8a4a"), at: { l: [0.12, 0.1], r: [0.88, 0.1] } }, 1.0, 0, -1.2, { delay: 0.2 });
  strings(scene, banner, ["l", "r"]);
  const lady = actor(scene, paintFairy(), 2.0, 2.9, -0.6);
  strings(scene, lady, ["head", "wand"]);
  const father = actor(scene, paintGeppetto("joy", "none"), 2.1, -2.3, 0.2);
  strings(scene, father, ["head", "hand"]);
  const wooden = toTexture(paintPinocchio({ mood: "joy", pose: "dance" }).canvas);
  const boy = toTexture(paintPinocchio({ mood: "joy", pose: "dance", real: true }).canvas);
  const hero = pinocchio(scene, { mood: "joy", pose: "dance" }, 1.8, 0.2, 0.8);
  strings(scene, hero, ["head", "handL", "handR"]);
  const cricket = actor(scene, paintCricket(), 0.7, 1.7, 1.4);
  const burst = stars(scene, "#ffe27a", () => new Vector3(0.2, FLOOR_Y + 1.2, 0.9));
  confetti(scene, 14);
  hearts(scene, -2.3, 2.3, 0.3);
  sparkles(scene, ctx.assets, [-4, 4, 0.4, 4.5], ["#ffe27a", "#ffd0f0", "#ffffff"]);
  let since = 0;
  let changed = false;
  scene.onUpdate((t, dt) => {
    if (scene.openness < 0.05) {
      since = 0;
      changed = false;
    } else since += dt;
    if (!changed && since > 2.2) {
      changed = true;
      burst.burst(40);
    }
    swapMap(hero.mesh, changed ? boy : wooden);
    hero.setNose(changed ? 0.5 : 1);
    hero.nose.material.color.set(changed ? "#f6c7a2" : "#d9965a");
    banner.mesh.position.y = 2.6 + Math.sin(t * 1.4) * 0.08;
    banner.mesh.rotation.z = Math.sin(t * 1.1) * 0.04;
    lady.mesh.position.y = 0.6 + Math.sin(t * 1.2) * 0.15;
    dangle(hero.mesh, t, changed ? 4.5 : 1.2, changed ? 0.45 : 0.05);
    dangle(father.mesh, t, 3.2, 0.25, 1);
    dangle(cricket.mesh, t, 6, 0.15, 2);
    scene.focus.set(0, 1.5, 0.4);
  });
  return scene;
};
