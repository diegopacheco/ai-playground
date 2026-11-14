import { alice, bubble, butterfly, rabbit, sister } from "../art/characters";
import { bankLayer, deepGlow, earthRing, frame, hedgeLayer, riverLayer, sky } from "../art/sets";
import { loop, smooth } from "../engine/timeline";
import { Tunnel } from "../engine/tunnel";
import { book, card, clock, FLOOR, jar, type SceneBuilder } from "./kit";
import { HOLE_X, RABBIT_LOOP, rabbitDive, rabbitX } from "./timelines";

function butterflies(t: Tunnel, count: number): void {
  const colors = ["#74c0fc", "#ffa94d", "#f783ac"];
  const items = Array.from({ length: count }, (_, i) => t.cutout(butterfly(0, colors[i % 3] ?? "#74c0fc"), 0.32, 0, 0, -2.2 - i * 0.5));
  t.onUpdate((time) => {
    items.forEach((b, i) => {
      b.position.set(Math.sin(time * 0.4 + i * 2) * 3.5, 0.6 + Math.sin(time * 0.9 + i) * 1.2, 0);
      b.scale.x = 0.35 + Math.abs(Math.sin(time * 9 + i)) * 0.65;
    });
  });
}

function riverbankSet(t: Tunnel, title: string): { girl: ReturnType<Tunnel["cutout"]>; sis: ReturnType<Tunnel["cutout"]> } {
  t.layer(frame({ paper: "#f6ead2", trim: "#c79a4a", shape: "oval", motif: "vines", title, seed: 3 }), 0);
  t.layer(bankLayer(true), -1.6);
  const sis = t.cutout(sister(), 2.2, -1.0, FLOOR + 0.25, -3.0);
  const girl = t.cutout(alice("sit"), 2.0, 0.7, FLOOR + 0.25, -2.8);
  t.layer(riverLayer(), -4.6);
  t.layer(sky("#8fd0ff", "#fdf3d8", 5), -7);
  butterflies(t, 3);
  return { girl, sis };
}

export const riverbank: SceneBuilder = (z) => {
  const t = new Tunnel(z, "#e9d3a6");
  const { girl, sis } = riverbankSet(t, "Alice's Adventures in Wonderland");
  t.onUpdate((time) => {
    girl.rotation.z = Math.sin(time * 0.8) * 0.03;
    girl.position.y = FLOOR + 0.25 + Math.abs(Math.sin(time * 0.5)) * 0.03;
    sis.rotation.z = Math.sin(time * 0.4 + 1) * 0.015;
  });
  return t;
};

export const rabbitScene: SceneBuilder = (z) => {
  const t = new Tunnel(z, "#d6e8c4");
  t.layer(frame({ paper: "#eef6e2", trim: "#5a9a4a", shape: "arch", motif: "vines", seed: 7 }), 0);
  t.layer(bankLayer(false), -1.4);
  const frames = [rabbit(0), rabbit(1)].map((c) => t.cutout(c, 2.2, 0, FLOOR + 0.3, -3.2));
  const words = t.cutout(bubble("Oh dear! I shall be late!", 640), 0.9, 0, 0.3, -2.6);
  const girl = t.cutout(alice("stand", "wow"), 2.4, 0, FLOOR + 0.2, -3.6);
  t.layer(hedgeLayer(true), -4.4);
  t.layer(sky("#a5d8ff", "#fff9db", 9), -7);
  butterflies(t, 2);
  t.onUpdate((time) => {
    const x = rabbitX(time);
    const dive = rabbitDive(time);
    frames.forEach((f, i) => {
      f.visible = Math.floor(time * 7) % 2 === i;
      f.position.set(x, FLOOR + 0.3 + Math.abs(Math.sin(time * 11)) * 0.25 * (1 - dive) - dive * 1.2, 0);
      f.scale.setScalar(1 - dive * 0.7);
    });
    const lt = loop(time, RABBIT_LOOP);
    words.position.set(Math.min(x, HOLE_X - 1.2) + 0.4, 0.4, 0);
    words.scale.setScalar(Math.max(0.0001, smooth(lt, 0.6, 1.1) * (1 - smooth(lt, 4.6, 5))));
    girl.position.x = Math.min(-5.5 + Math.max(0, lt - 1.8) * 1.3, HOLE_X - 1.6);
    girl.position.y = FLOOR + 0.2 + Math.abs(Math.sin(time * 6)) * 0.08;
  });
  return t;
};

const RINGS = 12;
const RUN = 16.5;
const SPEED = 1.5;

export const fall: SceneBuilder = (z) => {
  const t = new Tunnel(z, null);
  t.layer(deepGlow(), -7.5);
  const rings = Array.from({ length: RINGS }, (_, i) => t.layer(earthRing(i + 2), -7));
  const things = [clock(), jar(), book("#3b6fd8"), clock(), jar(), book("#c92a2a"), card(3), card(6)].map((object, i) => {
    const a = (i / 8) * Math.PI * 2;
    return { object: t.solid(object, Math.cos(a) * 2.6, Math.sin(a) * 1.7, -7), spin: 0.6 + (i % 3) * 0.4, offset: i * (RUN / 8) + 0.7 };
  });
  const girl = t.cutout(alice("fall"), 2.6, 0, -1.4, -0.6, false);
  t.onUpdate((time) => {
    rings.forEach((ring, i) => {
      const local = loop(i * (RUN / RINGS) + time * SPEED, RUN);
      ring.position.z = local;
      ring.rotation.z = i * 0.7 + local * 0.05;
      ring.userData.alpha = smooth(local, 0, 1.6);
    });
    for (const thing of things) {
      const local = loop(thing.offset + time * SPEED * 1.2, RUN);
      thing.object.position.z = local;
      thing.object.rotation.set(time * thing.spin, time * thing.spin * 0.7, time * 0.3);
      thing.object.visible = local > 0.4 && local < 14;
    }
    girl.position.set(Math.sin(time * 0.5) * 0.7, -1.4 + Math.sin(time * 0.8) * 0.25, 0);
    girl.rotation.z = Math.sin(time * 0.35) * 0.35;
  });
  return t;
};

export const waking: SceneBuilder = (z) => {
  const t = new Tunnel(z, "#e9d3a6");
  const { girl } = riverbankSet(t, "The End");
  const cards = Array.from({ length: 30 }, (_, i) => ({ mesh: t.solid(card(i), 0, 0, -3.4), seed: i }));
  const leaves = Array.from({ length: 10 }, (_, i) => t.cutout(butterfly(1, i % 2 === 0 ? "#e8892c" : "#c0512a"), 0.26, 0, 0, -2 - (i % 4) * 0.6));
  t.onUpdate((time) => {
    for (const { mesh, seed } of cards) {
      const life = loop(time * 0.18 + seed / 30, 1);
      const a = seed * 2.4 + time * 1.4 + life * 6;
      const r = 0.4 + life * 3.6;
      mesh.position.set(Math.cos(a) * r, FLOOR + 0.4 + life * 6.4, Math.sin(a) * r * 0.5);
      mesh.rotation.set(time * 2 + seed, time * 1.5 + seed, a);
    }
    leaves.forEach((leafPiece, i) => {
      const fallT = loop(time * 0.12 + i / 10, 1);
      leafPiece.position.set(-4 + ((i * 1.7) % 8) + Math.sin(time + i) * 0.5, 3.2 - fallT * 6.6, 0);
      leafPiece.rotation.z = Math.sin(time * 2 + i) * 0.8;
    });
    girl.rotation.z = Math.sin(time * 0.6) * 0.04;
  });
  return t;
};
