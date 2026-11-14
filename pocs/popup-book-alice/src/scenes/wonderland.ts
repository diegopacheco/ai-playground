import { Color, Mesh, MeshStandardMaterial, TorusGeometry } from "three";
import { alice, bubble, caterpillar, cardMan, cheshireBody, dormouse, flamingo, grin, hare, hatter, hedgehog, queen } from "../art/characters";
import { catBranch, curtains, wallpaper, forestLayer, frame, gardenBack, hallDoors, hareHouse, hoops, lawn, rose, roseBush, signpost, sky, tallFlowers, teaTable } from "../art/sets";
import type { Paper } from "../engine/tunnel";
import { loop, smooth } from "../engine/timeline";
import { Tunnel } from "../engine/tunnel";
import { bottle, cake, FLOOR, glassTable, mushroom, type SceneBuilder, teacup, teapot } from "./kit";
import { catBody, catGrin, growAlice, MOVE_EVERY, roseRed, SEATS, seatShift, shrinkRoom, shrinkSet } from "./timelines";

function paperOf(holder: ReturnType<Tunnel["cutout"]>): Paper {
  return holder.children[0] as Paper;
}

export const drinkMe: SceneBuilder = (z) => {
  const t = new Tunnel(z, "#c9a27a");
  t.pivot.set(1.5, FLOOR, -3.2);
  t.layer(frame({ paper: "#6b3e22", trim: "#e0b040", shape: "keyhole", motif: "wood", seed: 11 }), 0, false);
  t.layer(curtains("#8a2b4a"), -1.2);
  const table = t.solid(glassTable(), -0.9, FLOOR, -3);
  const potion = t.solid(bottle(), -1.2, FLOOR + 1.63, -3);
  t.layer(hallDoors(), -7);
  const girl = t.cutout(alice("stand", "wow"), 2.6, 1.5, FLOOR, -3.2, false);
  const label = t.cutout(bubble("Drink me!", 360), 0.7, -1.2, 0.2, -2.6);
  t.onUpdate((time) => {
    t.scaleSet(shrinkSet(time));
    potion.rotation.y = time * 0.6;
    table.rotation.y = Math.sin(time * 0.3) * 0.1;
    girl.rotation.z = Math.sin(time * 1.2) * 0.03;
    const lt = loop(time, 12);
    label.scale.setScalar(Math.max(0.0001, smooth(lt, 0.4, 0.9) * (1 - smooth(lt, 2.2, 2.6))));
  });
  return t;
};

export const eatMe: SceneBuilder = (z) => {
  const t = new Tunnel(z, "#c9a27a");
  t.pivot.set(0.6, FLOOR, -5.5);
  t.layer(frame({ paper: "#5a3a6a", trim: "#e0b040", shape: "rect", motif: "wood", seed: 13 }), 0, false);
  t.layer(curtains("#3b6fd8"), -1.2);
  const box = t.solid(cake(), -1.8, FLOOR, -2.6);
  t.layer(hallDoors(), -5.5);
  t.layer(wallpaper(), -7, false);
  const girl = t.cutout(alice("stand", "wow"), 2.6, 0.6, FLOOR, -2.3, false);
  const words = t.cutout(bubble("Curiouser and curiouser!", 640), 0.9, -1.6, 1.9, -2.2, false);
  t.onUpdate((time) => {
    t.scaleSet(shrinkRoom(time));
    girl.scale.setScalar(growAlice(time));
    girl.rotation.z = Math.sin(time * 1.5) * 0.02;
    box.rotation.y = Math.sin(time * 0.5) * 0.3;
    const lt = loop(time, 12);
    words.scale.setScalar(Math.max(0.0001, smooth(lt, 5.2, 5.8) * (1 - smooth(lt, 8.6, 9))));
  });
  return t;
};

export const caterpillarScene: SceneBuilder = (z) => {
  const t = new Tunnel(z, "#9ccf8a");
  t.layer(frame({ paper: "#dff0c8", trim: "#4c8a3a", shape: "arch", motif: "grass", seed: 17 }), 0);
  t.layer(tallFlowers(5, true), -1.5);
  const big = t.solid(mushroom(1.6, 1.5, "#b5653a"), 0.9, FLOOR, -3.8);
  t.solid(mushroom(0.6, 0.6, "#c2410c"), -2.2, FLOOR, -2.9);
  t.solid(mushroom(0.45, 0.4, "#a8744a"), 2.9, FLOOR, -2.6);
  const worm = t.cutout(caterpillar(), 2.0, 0.9, FLOOR + 2.3, -3.8);
  const girl = t.cutout(alice("stand"), 1.4, -1.0, FLOOR, -2.9);
  const words = t.cutout(bubble("Who are you?", 420), 0.75, 0.2, 1.7, -3.2);
  t.layer(tallFlowers(9, true), -5);
  t.layer(sky("#2a2050", "#c08ad8", 21, false, true), -7);
  const rings = Array.from({ length: 4 }, () => {
    const ring = new Mesh(new TorusGeometry(0.22, 0.05, 10, 30), new MeshStandardMaterial({ color: "#e8e0ff", transparent: true, opacity: 0.8, roughness: 1 }));
    t.solid(ring, 0, 0, -3.6);
    return ring;
  });
  t.onUpdate((time) => {
    worm.rotation.z = Math.sin(time * 0.9) * 0.04;
    worm.position.y = FLOOR + 2.3 + Math.sin(time * 1.8) * 0.04;
    big.rotation.y = time * 0.05;
    girl.position.y = FLOOR + Math.abs(Math.sin(time * 1.3)) * 0.06;
    rings.forEach((ring, i) => {
      const life = loop(time * 0.3 + i / 4, 1);
      ring.position.set(1.1 + life * 0.8, FLOOR + 4.1 + life * 2.4, 0);
      ring.scale.setScalar(0.6 + life * 2.2);
      (ring.material as MeshStandardMaterial).opacity = 0.8 * (1 - life);
    });
    const lt = loop(time, 9);
    words.scale.setScalar(Math.max(0.0001, smooth(lt, 1, 1.5) * (1 - smooth(lt, 6, 6.5))));
  });
  return t;
};

export const cheshire: SceneBuilder = (z) => {
  const t = new Tunnel(z, "#8a7aa8");
  t.layer(frame({ paper: "#2a2040", trim: "#8a7aa8", shape: "arch", motif: "forest", seed: 19 }), 0);
  t.cutout(signpost(), 2.8, -3.1, FLOOR, -1.8);
  const girl = t.cutout(alice("stand"), 2.4, -1.3, FLOOR, -2.4);
  t.layer(catBranch(), -3.2);
  const cat = t.cutout(cheshireBody(), 1.5, 0.9, 0.42, -3.15);
  const smile = t.cutout(grin(), 0.42, 0.38, 0.92, -3.1);
  t.layer(forestLayer(4, true), -4.8);
  t.layer(sky("#1b1638", "#5a4a8a", 25, false, true), -7);
  t.onUpdate((time) => {
    paperOf(cat).userData.alpha = catBody(time);
    paperOf(smile).userData.alpha = catGrin(time);
    smile.scale.setScalar(1 + (1 - catBody(time)) * 0.3);
    cat.rotation.z = Math.sin(time * 0.7) * 0.02;
    girl.rotation.z = Math.sin(time * 0.9) * 0.03;
  });
  return t;
};

export const teaParty: SceneBuilder = (z) => {
  const t = new Tunnel(z, "#f3c4d4");
  t.layer(frame({ paper: "#d6336c", trim: "#fffaf0", shape: "rect", motif: "tent", seed: 23 }), 0);
  const host = t.cutout(hatter(), 3.0, 2.4, FLOOR, -3.6);
  const rabbitHare = t.cutout(hare(), 2.8, -2.5, FLOOR, -3.6);
  const girl = t.cutout(alice("stand"), 2.3, 0.1, FLOOR, -3.7);
  t.cutout(teaTable(), 2.0, 0, FLOOR, -3.0);
  const pot = teapot();
  t.solid(pot.group, -0.9, FLOOR + 1.62, -2.95);
  const mouse = t.cutout(dormouse(), 0.55, -0.9, FLOOR + 1.9, -3.0);
  const colors = ["#f783ac", "#74c0fc", "#ffd43b", "#b197fc", "#8ce99a", "#ffa94d"];
  const cups = colors.map((c) => t.solid(teacup(c), 0, FLOOR + 1.62, -2.92));
  const words = t.cutout(bubble("No room! No room!", 520), 0.85, -2.1, 1.0, -3.2);
  t.layer(hareHouse(), -5.5);
  t.layer(sky("#ffd8a8", "#fff4e6", 29), -7);
  t.onUpdate((time) => {
    const shift = seatShift(time);
    cups.forEach((cup, i) => {
      const slot = (((i + shift) % SEATS) + SEATS) % SEATS;
      const x = -3.8 + slot * 1.35;
      const hopping = loop(time, MOVE_EVERY) > MOVE_EVERY - 0.8;
      cup.position.set(x, FLOOR + 1.62 + (hopping ? Math.sin((loop(time, MOVE_EVERY) - (MOVE_EVERY - 0.8)) / 0.8 * Math.PI) * 0.4 : 0), 0);
      cup.rotation.y = time * 0.4 + i;
    });
    const lid = Math.max(0, Math.sin(time * 0.9)) ** 3;
    pot.lid.position.y = 0.55 + lid * 0.3;
    mouse.position.y = FLOOR + 1.9 + lid * 0.5 - 0.5;
    paperOf(mouse).userData.alpha = lid > 0.15 ? 1 : 0;
    host.rotation.z = Math.sin(time * 2.2) * 0.05;
    rabbitHare.rotation.z = Math.sin(time * 2.6 + 1) * 0.05;
    girl.rotation.z = Math.sin(time) * 0.02;
    const lt = loop(time, MOVE_EVERY * 2);
    words.scale.setScalar(Math.max(0.0001, smooth(lt, MOVE_EVERY - 1.4, MOVE_EVERY - 1) * (1 - smooth(lt, MOVE_EVERY + 1, MOVE_EVERY + 1.4))));
  });
  return t;
};

export const croquet: SceneBuilder = (z) => {
  const t = new Tunnel(z, "#f2b8b8");
  t.layer(frame({ paper: "#fbe3e3", trim: "#c92a2a", shape: "arch", motif: "roses", seed: 29 }), 0);
  t.cutout(roseBush(), 2.6, 2.7, FLOOR, -2.1);
  const red = new Color("#e03131");
  const white = new Color("#ffffff");
  const roses = Array.from({ length: 7 }, (_, i) => {
    const r = t.cutout(rose(), 0.36, 1.8 + (i % 4) * 0.55, FLOOR + 0.8 + Math.floor(i / 4) * 0.6 + (i % 2) * 0.25, -2.05);
    return paperOf(r).material;
  });
  const painters = [cardMan(2, 0), cardMan(5, 1)].map((c, i) => t.cutout(c, 2.0, i === 0 ? 1.1 : 4.0, FLOOR, -1.8));
  painters[1]!.scale.x = -1;
  const majesty = t.cutout(queen(), 3.1, -2.9, FLOOR, -3.4);
  const shout = t.cutout(bubble("Off with their heads!", 600), 0.85, -2.1, 1.3, -3.0);
  const girl = t.cutout(alice("stand"), 2.3, -0.4, FLOOR, -3.0);
  const bird = t.cutout(flamingo(), 1.9, 0.6, FLOOR, -3.1);
  const ball = t.cutout(hedgehog(), 0.4, 0, FLOOR, -2.95);
  t.layer(hoops(), -4.4);
  t.layer(lawn(), -5.6);
  t.layer(gardenBack(), -7);
  t.onUpdate((time) => {
    roses.forEach((material, i) => material.color.copy(white).lerp(red, roseRed(time, i, roses.length)));
    painters.forEach((p, i) => (p.rotation.z = Math.sin(time * 5 + i) * 0.05));
    majesty.rotation.z = Math.sin(time * 3) * 0.04;
    const lt = loop(time, 8);
    shout.scale.setScalar(Math.max(0.0001, smooth(lt, 1, 1.4) * (1 - smooth(lt, 4.5, 5))));
    ball.position.x = -0.2 + smooth(lt, 2, 6) * 3.4;
    ball.rotation.z = -smooth(lt, 2, 6) * 12;
    bird.rotation.z = -0.25 * smooth(lt, 1.4, 2) * (1 - smooth(lt, 2, 2.6));
    girl.rotation.z = Math.sin(time) * 0.02;
  });
  return t;
};
