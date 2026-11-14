import { Vector3 } from "three";
import { bird, donkey, horse } from "../art/animals";
import { bigTree, city, cliff, ground, palm, scroll, titleBanner, woodpile } from "../art/scenery";
import { Act } from "../engine/act";
import { Sil } from "../engine/silhouette";
import { loop, smooth } from "../engine/timeline";
import { boulder, cast, flip, framesFor, type SceneBuilder, ShadowSpray, walkBob } from "./kit";
import { doorOpen, RIDE_LOOP, RIDERS, riderX, SESAME_LOOP, shoutOn } from "./timelines";

function flock(act: Act, count: number, y: number, depth: number, speed: number): void {
  const birds = Array.from({ length: count }, (_, i) => {
    const b = new Sil(framesFor(2, bird), 0.22);
    act.place(b, 0, depth + i * 0.08, { y: y + (i % 2) * 0.25, kind: "rise" });
    return b;
  });
  act.onUpdate((t) => {
    birds.forEach((b, i) => {
      b.root.position.x = 5 - loop(t * speed + i * 0.6, 10);
      b.root.position.y = Math.sin(t * 2 + i) * 0.08;
      b.root.scale.x = -1;
      b.show(Math.floor(t * 7 + i) % 2);
    });
  });
}

export const title: SceneBuilder = () => {
  const act = new Act().tint("#ffb347", "#b4232c");
  act.place(new Sil(city(2.0), 2.0, { density: 0.8 }), 0, 1.0);
  act.place(new Sil(palm(2.7, 1, 3), 2.7), -3.35, 0.25);
  act.place(flip(new Sil(palm(2.3, 1, 5), 2.3)), 3.45, 0.32);
  act.place(new Sil(ground("dunes"), 0.55), 0, 0.04, { y: -0.1 });
  const banner = act.place(new Sil(titleBanner(), 1.6), 0, 0.12, { y: 2.85, kind: "rise", delay: 0.35 });
  const ali = cast.ali();
  act.place(ali.body, 0, 0.16, { rod: true });
  const mule = new Sil(framesFor(2, (f) => donkey(f, "wood")), 1.0);
  act.place(mule, 0, 0.2, { rod: true });
  flock(act, 3, 2.1, 0.6, 0.8);
  act.lampMotion = (t, out) => out.set(Math.sin(t * 0.3) * 0.35, Math.sin(t * 0.5) * 0.12, 0);
  act.onUpdate((t) => {
    const x = -5 + loop(t * 0.5, 10.5);
    ali.body.root.position.x = x;
    walkBob(ali.body, t, 4.2);
    ali.arm.root.rotation.z = Math.sin(t * 4.2) * 0.25;
    mule.root.position.x = x - 1.3;
    mule.show(Math.floor(t * 4.2 / Math.PI) % 2);
    banner.root.position.y = Math.sin(t * 0.9) * 0.05;
    banner.root.rotation.z = Math.sin(t * 0.6) * 0.012;
  });
  return act;
};

export const woodcutter: SceneBuilder = (assets) => {
  const act = new Act().tint("#ffd27a", "#e8761c");
  act.place(new Sil(bigTree(3.7), 3.7), -2.5, 0.45);
  act.place(flip(new Sil(palm(2.6, 1, 8), 2.6)), 3.5, 0.3);
  act.place(new Sil(palm(1.9, -1, 9), 1.9), 2.6, 0.7);
  act.place(new Sil(ground("hills"), 0.55), 0, 0.04, { y: -0.1 });
  act.place(new Sil(woodpile(), 0.55), 0.55, 0.1);
  const ali = cast.ali("axe");
  act.place(ali.body, -0.45, 0.14, { rod: true });
  const donkeys = [1.9, 3.0].map((x, i) => {
    const d = new Sil(framesFor(2, (f) => donkey(f, "wood")), 1.0);
    act.place(i === 0 ? d : flip(d), x, 0.2 + i * 0.1, { rod: true });
    return d;
  });
  flock(act, 4, 2.7, 0.8, 0.6);
  const chips = new ShadowSpray(60, null, 1, 0.4, 6).addTo(act);
  let last = 0;
  act.onUpdate((t, dt) => {
    const c = loop(t, 1.5) / 1.5;
    const swing = c < 0.55 ? smooth(c, 0, 0.55) : 1 - smooth(c, 0.55, 0.68);
    ali.arm.root.rotation.z = -0.3 + swing * 2.9;
    ali.body.root.rotation.z = -0.04 + swing * 0.06;
    if (c >= 0.66 && last < 0.66) {
      for (let i = 0; i < 9; i++) chips.emit(new Vector3(0.5, 0.55, -0.12), new Vector3((Math.random() - 0.2) * 2.4, 1.5 + Math.random() * 2, 0), 0.9, 0.05 + Math.random() * 0.04, 14);
    }
    last = c;
    chips.update(dt);
    donkeys.forEach((d, i) => {
      d.root.position.y = Math.abs(Math.sin(t * 1.3 + i)) * 0.02;
      d.root.rotation.z = Math.sin(t * 0.8 + i * 2) * 0.015;
    });
  });
  void assets;
  return act;
};

export const riders: SceneBuilder = (assets) => {
  const act = new Act().tint("#ffb35c", "#8f2d0b");
  act.place(new Sil(ground("hills"), 0.55), 0, 0.04, { y: -0.1 });
  act.place(new Sil(bigTree(3.6), 3.6), 2.75, 0.22);
  const ali = cast.ali();
  ali.body.root.scale.setScalar(0.62);
  act.place(flip(ali.body), 2.25, 0.24, { y: 2.05 });
  const depths = [0.12, 0.3, 0.55, 0.85, 1.15, 0.2, 0.45, 1.35];
  const horses = depths.slice(0, RIDERS).map((depth) => {
    const h = flip(new Sil(framesFor(2, (f) => horse(f, true)), 1.9));
    act.place(h, 0, depth, { y: depth * 0.25 });
    return h;
  });
  const dust = new ShadowSpray(120, assets.disc, 0.22, 2.4, -0.25).addTo(act);
  act.onUpdate((t, dt) => {
    horses.forEach((h, i) => {
      const x = riderX(t, i);
      h.root.position.x = x;
      h.root.position.y = Math.abs(Math.sin(t * 9 + i)) * 0.07;
      h.show(Math.floor(t * 8 + i) % 2);
      if (Math.random() < dt * 9 && Math.abs(x) < 5) dust.emit(new Vector3(x + 0.6, 0.25, -(depths[i] ?? 0.2) - 0.02), new Vector3(0.6 + Math.random() * 0.6, 0.15 + Math.random() * 0.3, 0), 1.6, 0.5 + Math.random() * 0.4);
    });
    dust.update(dt, 1.6);
    ali.body.root.position.y = Math.max(0, Math.sin(t * 0.7)) * 0.2;
  });
  act.lampMotion = (t, out) => out.set(Math.sin(t * 31) * 0.03, Math.sin(t * 27) * 0.03, Math.sin((t / RIDE_LOOP) * Math.PI * 2) * 0.4);
  return act;
};

const THIEF_HOMES = [-2.85, -3.65];

export const sesame: SceneBuilder = (assets) => {
  const act = new Act().tint("#7ef0dc", "#0d5752");
  act.place(new Sil(ground("hills"), 0.55), 0, 0.04, { y: -0.1 });
  act.place(new Sil(cliff(3.5), 3.5), 0.85, 0.35);
  const rock = boulder(0.6);
  act.solid(rock, 0.86, 0.55, 0.62);
  const captain = cast.captain();
  act.place(captain.body, -1.75, 0.14, { rod: true });
  const thieves = THIEF_HOMES.map((x, i) => {
    const th = cast.thief();
    act.place(th.body, x, 0.24 + i * 0.16, { rod: true });
    return th;
  });
  const words = new Sil(scroll("Open, Sesame!"), 0.62);
  act.place(words, -1.6, 0.08, { y: 2.55, kind: "rise" });
  const dust = new ShadowSpray(60, assets.disc, 0.25, 2, -0.2).addTo(act);
  act.onUpdate((t, dt) => {
    const shout = shoutOn(t);
    const open = doorOpen(t);
    captain.arm.root.rotation.z += ((shout ? 2.5 : 0.1) - captain.arm.root.rotation.z) * Math.min(dt * 6, 1);
    captain.body.root.rotation.z = shout ? Math.sin(t * 20) * 0.01 : 0;
    words.root.scale.setScalar(shout ? Math.min(1, words.root.scale.x + dt * 4) : Math.max(0.0001, words.root.scale.x - dt * 4));
    rock.position.x = open * 1.45;
    rock.rotation.z = -open * 2.6;
    if (open > 0.02 && open < 0.98 && Math.random() < dt * 20) dust.emit(new Vector3(0.86 + open * 1.45, 0.15, -0.5), new Vector3(Math.random() - 0.3, 0.3, 0), 1.4, 0.4);
    dust.update(dt, 1.4);
    const lt = loop(t, SESAME_LOOP);
    thieves.forEach((th, i) => {
      const home = THIEF_HOMES[i] ?? 0;
      th.body.root.position.x = -2.2 * (1 - smooth(lt, 0, 1.6)) + smooth(lt, 5.6 + i * 0.5, 9.6 + i * 0.5) * (1.1 - home);
      walkBob(th.body, t, 3.4, 0.035);
    });
  });
  act.lampMotion = (t, out) => out.set(0, 0.15 * smooth(loop(t, SESAME_LOOP), 1.5, 3), 1.15 * (smooth(loop(t, SESAME_LOOP), 1.5, 3.2) - smooth(loop(t, SESAME_LOOP), 6, 9)));
  act.levelAt = (t) => 1 + doorOpen(t) * 0.15;
  return act;
};
