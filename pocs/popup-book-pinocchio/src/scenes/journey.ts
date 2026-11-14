import { CircleGeometry, CylinderGeometry, Group, Mesh, MeshStandardMaterial, SphereGeometry, Vector3 } from "three";
import { toTexture } from "../art/canvas";
import { paintBubble, paintBush, paintSack, paintSign } from "../art/props";
import { paintCat, paintFairy, paintFireEater, paintFox, paintPinocchio, paintPuppet, paintWoodpecker } from "../art/puppets";
import { FLOOR_Y, cutoutMesh } from "../engine/popup";
import { coin, miniTheater } from "./models";
import { type SceneBuilder, actor, coinBurst, createScene, dangle, loop, pinocchio, pop, smooth, sparkles, strings } from "./kit";
import {
  FIELD_LOOP,
  HOLE_X,
  NOSE_LOOP,
  SNEEZE_LOOP,
  birdsPerched,
  buried,
  coinsStolen,
  dreamTree,
  foxX,
  noseLength,
  sneezeSwell,
  sneezing,
} from "./timelines";

export const puppetShow: SceneBuilder = (ctx) => {
  const scene = createScene("tent", { key: "#ffe7c0", fill: "#ff8a8a", ambient: 0.55 });
  scene.solid(miniTheater(), -0.6, -1.3, 0.1);
  const minis = (["harlequin", "punch"] as const).map((kind, i) => {
    const mini = actor(scene, paintPuppet(kind), 0.95, -1.05 + i * 0.9, -1.1, { delay: 0.35 });
    strings(scene, mini, ["head"]);
    return mini;
  });
  const calm = toTexture(paintFireEater(false).canvas);
  const achoo = toTexture(paintFireEater(true).canvas);
  const giant = actor(scene, paintFireEater(false), 3.2, 2.6, -0.2);
  const bubble = scene.cutout(paintBubble("AH-CHOO!"), 0.9, 1.4, 0.4, { delay: 0.6 });
  const hero = pinocchio(scene, { mood: "joy", pose: "dance" }, 1.6, -2.6, 1.0);
  strings(scene, hero, ["head", "handL", "handR"]);
  const coins = coinBurst(scene, 2.1, 2.6, 0.2);
  let sneezed = false;
  sparkles(scene, ctx.assets, [-2, 0.8, 0.8, 2.4], ["#ffe27a", "#ff9fd0"], 6);
  scene.onUpdate((t) => {
    const c = loop(t, SNEEZE_LOOP);
    minis.forEach((mini, i) => {
      mini.mesh.position.y = 0.55 + Math.abs(Math.sin(t * 4 + i * 1.5)) * 0.3;
      mini.mesh.rotation.z = Math.sin(t * 4 + i * 1.5) * 0.25;
    });
    const swell = sneezeSwell(c);
    giant.mesh.scale.set(1 + swell * 0.08, 1 + swell * 0.05, 1);
    giant.mesh.rotation.z = swell * 0.08 - (sneezing(c) ? Math.max(0, 0.25 - (c - 3.2) * 0.4) : 0);
    const map = sneezing(c) || swell > 0.5 ? achoo : calm;
    if (giant.mesh.material.map !== map) {
      giant.mesh.material.map = map;
      giant.mesh.material.needsUpdate = true;
    }
    if (sneezing(c) && !sneezed) coins.burst(18);
    sneezed = sneezing(c);
    pop(bubble, sneezing(c) ? Math.min((c - 3.2) * 6, 1) : 0);
    bubble.position.y = 2.6;
    dangle(hero.mesh, t, sneezing(c) ? 7 : 2.5, sneezing(c) ? 0.5 : 0.15);
    scene.focus.set(0.2, 1.6, 0);
  });
  return scene;
};

export const foxCat: SceneBuilder = () => {
  const scene = createScene("road", { key: "#ffd09a", fill: "#ff8a5a", ambient: 0.5 });
  scene.cutout(paintSign("Miracles"), 1.5, -3.4, -1.0);
  scene.cutout(paintBush("#5a8a3a"), 0.6, 3.5, -1.2);
  scene.cutout(paintBush("#4a7a30"), 0.5, -3.6, 1.4);
  const fox = actor(scene, paintFox(), 2.1, -1.8, 0.1);
  const cat = actor(scene, paintCat(), 1.7, -0.5, -0.3);
  strings(scene, fox, ["head"]);
  strings(scene, cat, ["head"]);
  const hero = pinocchio(scene, { mood: "joy", pose: "carry" }, 1.7, 2.0, 0.6, { flip: true });
  strings(scene, hero, ["head", "handL", "handR"]);
  const coins = Array.from({ length: 5 }, () => {
    const piece = coin();
    scene.free(piece);
    return piece;
  });
  const hand = new Vector3();
  scene.onUpdate((t) => {
    const step = Math.abs(Math.sin(t * 2.2));
    fox.mesh.rotation.z = -0.08 + step * 0.14;
    fox.mesh.position.y = step * 0.08;
    fox.mesh.position.x = Math.sin(t * 0.5) * 0.3;
    cat.mesh.rotation.z = Math.sin(t * 1.6) * 0.07;
    cat.mesh.position.x = Math.sin(t * 0.5 + 0.6) * 0.3;
    dangle(hero.mesh, t, 2.4, 0.08);
    hero.mesh.updateWorldMatrix(true, false);
    hand.copy(hero.point("handR")).applyMatrix4(hero.mesh.matrixWorld);
    coins.forEach((piece, i) => {
      const a = t * 2 + (i / coins.length) * Math.PI * 2;
      piece.visible = scene.openness > 0.9;
      piece.position.set(hand.x + Math.cos(a) * 0.35, hand.y + 0.35 + Math.sin(a * 2) * 0.08, hand.z + Math.sin(a) * 0.35);
      piece.rotation.y = t * 4 + i;
    });
    scene.focus.set(0, 1.4, 0.2);
  });
  return scene;
};

function coinTree(): Group {
  const group = new Group();
  const trunk = new Mesh(new CylinderGeometry(0.08, 0.14, 1.4, 10), new MeshStandardMaterial({ color: "#c9932f", metalness: 0.6, roughness: 0.3 }));
  trunk.position.y = 0.7;
  group.add(trunk);
  const crown = new Mesh(new SphereGeometry(0.75, 20, 14), new MeshStandardMaterial({ color: "#5aa84a", roughness: 0.7, transparent: true, opacity: 0.8 }));
  crown.position.y = 1.7;
  group.add(crown);
  for (let i = 0; i < 22; i++) {
    const piece = coin();
    const a = i * 2.4;
    const h = (i / 22) * 2 - 1;
    const r = Math.sqrt(1 - h * h) * 0.78;
    piece.position.set(Math.cos(a) * r, 1.7 + h * 0.7, Math.sin(a) * r);
    piece.lookAt(piece.position.x * 2, piece.position.y, piece.position.z * 2);
    group.add(piece);
  }
  return group;
}

export const field: SceneBuilder = (ctx) => {
  const scene = createScene("field", { key: "#c8c8ff", fill: "#8a7aff", ambient: 0.3 });
  const hole = new Mesh(new CircleGeometry(0.32, 24), new MeshStandardMaterial({ color: "#1a0f08", roughness: 1 }));
  hole.rotation.x = -Math.PI / 2;
  hole.position.set(HOLE_X, FLOOR_Y + 0.01, 0.6);
  scene.free(hole);
  const tree = coinTree();
  tree.position.set(HOLE_X, FLOOR_Y, -0.6);
  scene.free(tree);
  const seeds = Array.from({ length: 5 }, (_, i) => {
    const piece = coin();
    piece.position.set(HOLE_X + (i - 2) * 0.08, FLOOR_Y + 1.6, 0.6);
    scene.free(piece);
    return piece;
  });
  const dreaming = toTexture(paintPinocchio({ mood: "sleep", pose: "stand" }).canvas);
  const awake = toTexture(paintPinocchio({ mood: "joy", pose: "stand" }).canvas);
  const robbed = toTexture(paintPinocchio({ mood: "scared", pose: "ears" }).canvas);
  const hero = pinocchio(scene, { mood: "joy", pose: "stand" }, 1.6, HOLE_X - 1.1, 0.9);
  strings(scene, hero, ["head", "handL", "handR"]);
  const fox = actor(scene, paintFox(), 1.9, 0, 0.2);
  const cat = actor(scene, paintCat(), 1.5, 0, -0.1);
  const sack = cutoutMesh(paintSack(), 0.55);
  sack.position.set(0.5, 0.5, 0.05);
  fox.mesh.add(sack);
  const dream = sparkles(scene, ctx.assets, [HOLE_X - 1, HOLE_X + 1, 1, 3], ["#ffe27a", "#fff3a0"], 0);
  scene.onUpdate((t) => {
    const c = loop(t, FIELD_LOOP);
    seeds.forEach((piece, i) => {
      const fall = smooth(c, 0.6 + i * 0.25, 1.0 + i * 0.25);
      piece.position.y = FLOOR_Y + 1.6 - fall * 1.55;
      piece.rotation.y = t * 5;
      piece.visible = !buried(c) && scene.openness > 0.9;
    });
    const grow = dreamTree(c);
    tree.scale.setScalar(Math.max(grow, 0.001));
    tree.visible = grow > 0.01 && scene.openness > 0.9;
    tree.rotation.y = t * 0.3;
    dream.emitting = grow > 0.5;
    const map = coinsStolen(c) ? robbed : grow > 0.05 ? dreaming : awake;
    if (hero.mesh.material.map !== map) {
      hero.mesh.material.map = map;
      hero.mesh.material.needsUpdate = true;
    }
    hero.mesh.rotation.z = c < 2.2 ? Math.sin(t * 8) * 0.12 : coinsStolen(c) ? Math.sin(t * 20) * 0.05 : 0;
    const fx = foxX(c);
    fox.mesh.position.x = fx;
    cat.mesh.position.x = fx + 0.9;
    const fleeing = c > 9.8;
    fox.mesh.scale.x = fleeing ? 1 : -1;
    cat.mesh.scale.x = fleeing ? 1 : -1;
    const tiptoe = Math.abs(Math.sin(t * (fleeing ? 12 : 4)));
    fox.mesh.position.y = tiptoe * 0.1;
    cat.mesh.position.y = Math.abs(Math.sin(t * (fleeing ? 12 : 4) + 1)) * 0.1;
    sack.visible = coinsStolen(c);
    scene.focus.set(Math.min(fx, 2), 1.3, 0.4);
  });
  return scene;
};

export const fairy: SceneBuilder = (ctx) => {
  const scene = createScene("grove", { key: "#bfe8ff", fill: "#7ad7ff", ambient: 0.4 });
  const lady = actor(scene, paintFairy(), 2.5, -2.2, -0.3);
  strings(scene, lady, ["head", "wand"]);
  const calm = toTexture(paintPinocchio({ mood: "smile", pose: "stand" }).canvas);
  const scared = toTexture(paintPinocchio({ mood: "scared", pose: "stand" }).canvas);
  const hero = pinocchio(scene, { mood: "smile", pose: "stand" }, 1.8, 1.4, 0.5, { flip: true });
  strings(scene, hero, ["head", "handL", "handR"]);
  const birds = [0, 1, 2].map(() => {
    const bird = cutoutMesh(paintWoodpecker(), 0.34);
    bird.visible = false;
    scene.free(bird);
    return bird;
  });
  sparkles(scene, ctx.assets, [-3.2, -1, 1.2, 3.2], ["#7affff", "#ffffff", "#bfe8ff"], 10);
  const base = new Vector3();
  const tip = new Vector3();
  scene.onUpdate((t) => {
    const c = loop(t, NOSE_LOOP);
    const length = noseLength(c);
    hero.setNose(length);
    const map = length > 1 ? scared : calm;
    if (hero.mesh.material.map !== map) {
      hero.mesh.material.map = map;
      hero.mesh.material.needsUpdate = true;
    }
    lady.mesh.position.y = 0.5 + Math.sin(t * 1.2) * 0.15;
    lady.mesh.rotation.z = Math.sin(t * 0.9) * 0.05;
    hero.mesh.rotation.z = length > 1 ? Math.sin(t * 3) * 0.03 : 0;
    hero.nose.updateWorldMatrix(true, false);
    base.set(0, 0, 0).applyMatrix4(hero.nose.matrixWorld);
    tip.set(1, 0, 0).applyMatrix4(hero.nose.matrixWorld);
    const perched = birdsPerched(c);
    birds.forEach((bird, i) => {
      bird.visible = scene.openness > 0.9;
      if (perched) {
        const along = 0.35 + i * 0.25;
        bird.position.lerpVectors(base, tip, along);
        bird.position.y += 0.02 + Math.abs(Math.sin(t * 10 + i)) * 0.05;
        bird.rotation.z = Math.sin(t * 10 + i) * 0.2;
        bird.scale.x = -1;
      } else {
        const a = t * 1.4 + i * 2.1;
        bird.position.set(0.6 + Math.cos(a) * 1.4, FLOOR_Y + 3.1 + Math.sin(a * 2) * 0.3, 0.4 + Math.sin(a) * 0.6);
        bird.rotation.z = 0;
        bird.scale.x = Math.sin(a) > 0 ? 1 : -1;
      }
    });
    scene.focus.set(0, 1.6, 0.3);
  });
  return scene;
};
