import { PointLight } from "three";
import { paintWitch } from "../art/characters";
import { paintBubble, paintFlame, paintKitchenWall } from "../art/props";
import { type PopupScene, cutoutMesh } from "../engine/popup";
import { cage, candyCane, candyHouse, cauldron, gumdrop, logsFire, lollipop } from "./models";
import { type SceneBuilder, bob, butterflies, child, createScene, faces, loop, particles, pop, puffTexture, setFace, sky, smooth, sparkles, trees } from "./kit";

const SWEETS = ["#ff5d8f", "#ffd23f", "#7ad7ff", "#8fe36a", "#b38cff"];

function candyGarden(scene: PopupScene): void {
  scene.solid(lollipop("#ff5d8f", "#ffffff"), -2.6, -0.4, 0.4);
  scene.solid(lollipop("#8fe36a", "#ffffff", 0.7), 2.9, -0.2, 0.45);
  scene.solid(lollipop("#b38cff", "#ffd23f", 0.6), 3.3, 1.4, 0.6);
  scene.solid(candyCane(0.9), -3.3, 0.6, 0.5);
  scene.solid(candyCane(0.7, "#2f8f46"), 2.2, 1.9, 0.6);
  [[-1.8, 1.9], [-0.6, 2.2], [0.9, 2.0], [-2.9, 2.0], [1.9, 0.9]].forEach(([x, z], i) => scene.solid(gumdrop(SWEETS[i % SWEETS.length] ?? "#ff5d8f", 0.12), x ?? 0, z ?? 0, 0.65));
}

export function fireLight(scene: PopupScene, x: number, y: number, z: number): PointLight {
  const light = new PointLight("#ff8a3a", 14, 6, 1.6);
  light.position.set(x, y, z);
  scene.free(light);
  return light;
}

export function flames(scene: PopupScene, spots: [number, number, number][], height: number) {
  const canvas = paintFlame();
  return spots.map(([x, y, z]) => {
    const flame = cutoutMesh(canvas, height);
    flame.castShadow = false;
    flame.material.emissive.set("#ff7a1a");
    flame.material.emissiveMap = flame.material.map;
    flame.material.emissiveIntensity = 1.5;
    flame.position.set(x, y, z);
    scene.free(flame);
    return flame;
  });
}

function kitchen(scene: PopupScene): void {
  scene.cutout(paintKitchenWall(), 3.3, 0, -2.75, { delay: 0 });
}

export const candyHouseScene: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "meadow");
  sky(scene, "day");
  trees(scene, [[-3.4, -1.7, 2.2], [3.5, -1.8, 2.3]]);
  const house = candyHouse();
  house.group.scale.setScalar(1.15);
  scene.solid(house.group, 0, -1.0, 0.15);
  candyGarden(scene);
  const kids = [scene.cutout(child("joao", "joy", { prop: "bread" }), 1.15, -1.3, 0.9), scene.cutout(child("maria", "joy", { wave: true }), 1.1, 1.2, 1.0)];
  butterflies(scene, 2);
  sparkles(scene, ctx.assets, [-1.5, 1.5, 0.6, 3.0], ["#ffd0f0", "#fff3b0", "#b0f0ff"]);
  scene.onUpdate((t) => {
    kids.forEach((k, i) => bob(k, 4.5, 0.3, i * 1.4)(t));
    house.roof.rotation.y = Math.sin(t * 0.8) * 0.02;
  });
  return scene;
};

export const witchScene: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "meadow");
  sky(scene, "day");
  trees(scene, [[-3.5, -1.7, 2.2]]);
  const house = candyHouse();
  scene.solid(house.group, 1.2, -1.1, 0.15, -0.25);
  scene.solid(lollipop("#ff5d8f", "#ffffff"), 3.0, -0.3, 0.4);
  scene.solid(candyCane(0.8), -0.6, -1.0, 0.4);
  const looks = faces({ sweet: paintWitch("sweet"), wicked: paintWitch("wicked") });
  const witch = scene.cutout(paintWitch("sweet"), 1.6, 1.25, -0.1, { delay: 0.5 });
  const kids = [scene.cutout(child("maria", "smile"), 1.1, -1.7, 0.9), scene.cutout(child("joao", "smile"), 1.15, -2.6, 1.1)];
  const witchSays = scene.cutout(paintBubble(["Entrem, queridos!", "Tem doces para todos!"], "right"), 1.05, 0.3, 0.2, { delay: 0.6 });
  witchSays.position.y = 1.85;
  const mariaSays = scene.cutout(paintBubble(["Que casa", "deliciosa!"], "left"), 0.95, -1.4, 0.9, { delay: 0.6 });
  mariaSays.position.y = 1.45;
  scene.onUpdate((t) => {
    const c = loop(t, 8);
    house.door.rotation.y = -1.7 * smooth(c, 0.1, 0.8) * (1 - smooth(c, 7.4, 7.9));
    pop(witchSays, smooth(c, 0.9, 1.3) * (1 - smooth(c, 3.4, 3.7)));
    pop(mariaSays, smooth(c, 3.8, 4.2) * (1 - smooth(c, 5.6, 5.9)));
    const wicked = c > 5.8 && c < 7.6;
    setFace(witch, looks, wicked ? "wicked" : "sweet");
    witch.rotation.z = wicked ? Math.sin(t * 10) * 0.05 : Math.sin(t * 1.5) * 0.04;
    kids.forEach((k, i) => {
      k.position.x = smooth(c, 0.8, 3.6) * 0.6;
      bob(k, 6, 0.06, i)(t);
    });
  });
  return scene;
};

export const cageScene: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "interior");
  kitchen(scene);
  scene.solid(cage(), -1.8, -0.7, 0.2);
  const joao = scene.cutout(child("joao", "sad"), 1.0, -1.8, -0.7, { delay: 0.25 });
  scene.solid(logsFire(), 1.5, -1.0, 0.3);
  const pot = cauldron();
  scene.solid(pot.group, 1.5, -1.0, 0.35);
  const light = fireLight(scene, 1.5, 0.9, -0.4);
  const fire = flames(scene, [[1.15, 0.36, -0.55], [1.85, 0.36, -0.55]], 0.4);
  particles(scene, {
    texture: puffTexture(),
    count: 30,
    rate: 7,
    grow: 2.4,
    colors: ["#d8ffcf", "#b8f0a8", "#ffffff"],
    spawn: (p, rand) => {
      p.sprite.position.set(1.5 + (rand() - 0.5) * 0.6, 1.25, -1.0);
      p.velocity.set((rand() - 0.5) * 0.25, 0.6 + rand() * 0.3, 0.1);
      p.life = 1.8 + rand();
      p.size = 0.22;
    },
  });
  const witch = scene.cutout(paintWitch("wicked"), 1.6, 2.7, -0.4);
  const maria = scene.cutout(child("maria", "sad", { prop: "broom" }), 1.1, 0, 0.9);
  const dust = particles(scene, {
    texture: puffTexture(),
    count: 30,
    rate: 0,
    grow: 1.5,
    colors: ["#d9c49a", "#f4e6c4"],
    spawn: (p, rand) => {
      p.sprite.position.set(maria.position.x + 0.35, 0.42, 1.05);
      p.velocity.set((rand() - 0.2) * 0.6, 0.2 + rand() * 0.2, 0.2);
      p.life = 0.9 + rand() * 0.5;
      p.size = 0.12;
    },
  });
  scene.onUpdate((t, dt) => {
    pot.water.position.y = 0.86 + Math.sin(t * 12) * 0.012;
    light.intensity = 13 + Math.sin(t * 21) * 2 + Math.sin(t * 7) * 1.5;
    fire.forEach((f, i) => f.scale.set(1 + Math.sin(t * 13 + i * 2) * 0.12, 1 + Math.sin(t * 17 + i) * 0.22, 1));
    witch.rotation.z = Math.sin(t * 2.4) * 0.1;
    maria.position.x = Math.sin(t * 0.6) * 1.1;
    maria.rotation.z = Math.sin(t * 6) * 0.08;
    if (Math.random() < dt * 8) dust.burst(1);
    joao.position.x = Math.sin(t * 30) * (loop(t, 4) < 0.6 ? 0.03 : 0);
  });
  return scene;
};

export const boneScene: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "interior");
  kitchen(scene);
  scene.solid(cage(), -1.2, -0.6, 0.2);
  const joao = scene.cutout(child("joao", "smile", { prop: "bone" }), 1.0, -1.25, -0.6, { delay: 0.25 });
  const witch = scene.cutout(paintWitch("squint"), 1.65, 0.75, -0.1);
  const maria = scene.cutout(child("maria", "joy"), 1.05, 2.8, 0.9);
  const light = fireLight(scene, 0, 1.6, 0.4);
  light.color.set("#ffd27a");
  const lines: [string[], "left" | "right", number, number, number, number][] = [
    [["Mostre o seu", "dedo, João!"], "right", 0.6, -0.1, 0.4, 2.7],
    [["Aqui está!"], "left", -1.0, -0.3, 2.9, 4.3],
    [["Ainda tão", "magrinho!"], "right", 0.6, -0.1, 4.5, 6.8],
  ];
  const bubbles = lines.map(([text, tail, x, z]) => {
    const bubble = scene.cutout(paintBubble(text, tail), 1.05, x, z, { delay: 0.6 });
    bubble.position.y = 1.85;
    return bubble;
  });
  scene.onUpdate((t) => {
    const c = loop(t, 7.2);
    bubbles.forEach((b, i) => {
      const [, , , , start, end] = lines[i] ?? [[], "left", 0, 0, 0, 0];
      pop(b, smooth(c, start, start + 0.35) * (1 - smooth(c, end - 0.3, end)));
    });
    witch.rotation.z = 0.12 + Math.sin(t * 1.3) * 0.06;
    witch.position.x = -Math.abs(Math.sin(t * 1.3)) * 0.15;
    joao.rotation.z = Math.sin(t * 3) * 0.05;
    bob(maria, 9, 0.06)(t);
    light.intensity = 6 + Math.sin(t * 9) * 0.8;
  });
  return scene;
};
