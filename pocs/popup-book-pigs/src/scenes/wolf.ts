import type { Mesh, MeshStandardMaterial, PlaneGeometry, Texture } from "three";
import { toTexture } from "../art/canvas";
import { type WolfMood, paintWolf } from "../art/characters";
import { paintBubble, paintBush, paintStrawBits } from "../art/props";
import type { PopupScene } from "../engine/popup";
import { type HouseParts, brickHouse, hayBale, stickHouse, strawHouse } from "./houses";
import { type SceneBuilder, bob, createScene, debris, loop, pig, sky, smooth, trees, wind } from "./kit";

type Cutout = Mesh<PlaneGeometry, MeshStandardMaterial>;

function wolfFaces(moods: WolfMood[]): Record<string, Texture> {
  return Object.fromEntries(moods.map((m) => [m, toTexture(paintWolf(m))]));
}

function setFace(wolf: Cutout, faces: Record<string, Texture>, mood: WolfMood): void {
  const face = faces[mood];
  if (face && wolf.material.map !== face) {
    wolf.material.map = face;
    wolf.material.needsUpdate = true;
  }
}

function pop(mesh: Cutout, amount: number): void {
  const s = Math.max(amount, 0.0001);
  mesh.scale.set(s, s, s);
  mesh.visible = amount > 0.01;
}

export const knock: SceneBuilder = (ctx) => {
  const scene = createScene(ctx, "meadow");
  sky(scene, "day");
  trees(scene, [[3.4, -1.6, 2.2], [-3.4, -1.7, 2.0]]);
  const house = strawHouse();
  scene.solid(house.group, 1.0, -0.7, 0.15, -0.3);
  scene.solid(hayBale(), 2.7, 0.6, 0.5);
  const wolf = scene.cutout(paintWolf("sly"), 1.95, -1.4, 0.2);
  const scared = scene.cutout(pig("straw", "scared"), 0.85, 2.1, -0.1);
  const wolfSays = scene.cutout(paintBubble(["Little pig, little pig,", "let me come in!"], "left"), 0.95, -0.9, 0.3, { delay: 0.6 });
  wolfSays.position.y = 1.9;
  const pigSays = scene.cutout(paintBubble(["Not by the hair of my", "chinny chin chin!"], "right"), 0.95, 1.6, 0.4, { delay: 0.6 });
  pigSays.position.y = 1.55;
  scene.onUpdate((t) => {
    const c = loop(t, 6);
    pop(wolfSays, smooth(c, 0.2, 0.6) * (1 - smooth(c, 2.8, 3.1)));
    pop(pigSays, smooth(c, 3.2, 3.6) * (1 - smooth(c, 5.7, 6)));
    wolf.rotation.z = Math.sin(t * 1.5) * 0.06 - 0.04;
    scared.position.x = Math.sin(t * 40) * 0.02;
    house.walls.rotation.y = c > 0.4 && c < 1.2 ? Math.sin(t * 50) * 0.02 : 0;
  });
  return scene;
};

interface BlowSetup {
  house: HouseParts;
  bits: HTMLCanvasElement;
  runners: Cutout[];
  sturdy: boolean;
}

function blowLoop(scene: PopupScene, wolf: Cutout, faces: Record<string, Texture>, setup: BlowSetup): void {
  const gust = wind(scene, -1.5, 1.75, 0.3, 1);
  const flying = debris(scene, toTexture(setup.bits), 1.0, 0.4, -0.5, 1);
  const { house, runners, sturdy } = setup;
  const roofHome = house.roof.position.clone();
  const runnerHome = runners.map((r) => r.position.x);
  scene.onUpdate((t) => {
    const c = loop(t, 7.5);
    const inhale = smooth(c, 0, 1.4);
    const blowing = c > 1.4 && c < 4.2;
    const blowAmount = smooth(c, 1.4, 1.8) * (1 - smooth(c, 3.9, 4.2));
    const scale = 1 + inhale * 0.12 * (1 - smooth(c, 1.4, 1.7));
    wolf.scale.set(scale, scale, 1);
    gust.emitting = blowing;
    if (sturdy) {
      setFace(wolf, faces, c > 4.2 ? "tired" : blowing || c > 0.8 ? "puff" : "sly");
      house.group.rotation.z = blowAmount * Math.sin(t * 60) * 0.006;
      flying.emitting = false;
      return;
    }
    setFace(wolf, faces, blowing || c > 0.8 ? "puff" : "sly");
    const fly = smooth(c, 2.2, 3.4) * (1 - smooth(c, 6.8, 7.4));
    const shake = blowAmount * (1 - fly) * Math.sin(t * 45) * 0.08;
    house.walls.rotation.z = shake - fly * 0.4;
    house.walls.scale.y = 1 - fly * 0.85;
    house.roof.position.set(roofHome.x + fly * 3.2, roofHome.y + Math.sin(fly * Math.PI) * 1.6 - fly * 1.2, roofHome.z + fly * 0.8);
    house.roof.rotation.z = -fly * 4;
    house.roof.scale.setScalar(1 - fly * 0.5);
    flying.emitting = c > 1.8 && c < 3.6;
    runners.forEach((runner, i) => {
      const run = smooth(c, 2.4 + i * 0.2, 4.6) * (1 - smooth(c, 6.8, 7.4));
      runner.position.x = runnerHome[i]! + run * 2.6;
      runner.position.y = Math.abs(Math.sin(t * 14 + i)) * 0.15 * Math.sign(run);
      runner.visible = run < 0.98;
    });
  });
}

function blowScene(ctx: Parameters<SceneBuilder>[0], mood: "day" | "dusk", build: () => HouseParts, bits: HTMLCanvasElement, runnerNames: ("straw" | "sticks")[], sturdy: boolean): PopupScene {
  const scene = createScene(ctx, mood === "day" ? "meadow" : "dusk");
  sky(scene, mood);
  trees(scene, [[-3.4, -1.6, 2.1], [3.5, -1.8, 2.0]], mood === "dusk");
  const house = build();
  scene.solid(house.group, 1.0, -0.5, 0.15);
  const faces = wolfFaces(["sly", "puff", "tired"]);
  const wolf = scene.cutout(paintWolf("sly"), 2.4, -2.5, 0.3);
  const runners = runnerNames.map((name, i) => scene.cutout(pig(name, "scared"), 1.0, 1.9 + i * 0.6, 0.7 + i * 0.5));
  scene.cutout(paintBush(4), 0.5, -0.6, 1.8);
  blowLoop(scene, wolf, faces, { house, bits, runners, sturdy });
  return scene;
}

export const blowStraw: SceneBuilder = (ctx) => blowScene(ctx, "day", strawHouse, paintStrawBits("#f2cf6b", false), ["straw"], false);

export const blowSticks: SceneBuilder = (ctx) => blowScene(ctx, "dusk", stickHouse, paintStrawBits("#9c6b43", true), ["straw", "sticks"], false);

export const brickStands: SceneBuilder = (ctx) => {
  const scene = blowScene(ctx, "dusk", () => brickHouse(ctx.assets.bricks), paintStrawBits("#9c6b43", true), [], true);
  const cheer = (["straw", "sticks", "bricks"] as const).map((name, i) => scene.cutout(pig(name, "joy", true), 0.85, 2.4 + i * 0.55, 0.9 + (i % 2) * 0.4));
  scene.onUpdate((t) => cheer.forEach((p, i) => bob(p, 6, 0.2, i)(t)));
  return scene;
};
