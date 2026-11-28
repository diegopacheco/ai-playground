import * as THREE from "three";
import type { FurnitureItem } from "../types";
import { clayMaterial, glossyMaterial, glowMaterial } from "./materials";

const STYLE_ACCENTS: readonly string[] = ["#ffe066", "#ff8fab", "#72efdd"];

export function buildFurnitureMesh(item: FurnitureItem): THREE.Group {
  const group = new THREE.Group();
  const mainMat = clayMaterial(item.color, 0.56);
  const accentHex = STYLE_ACCENTS[item.styleVariant] ?? item.accentColor;
  const accentMat = clayMaterial(accentHex, 0.52);
  const creamMat = clayMaterial("#fff8ee", 0.6);
  const darkMat = glossyMaterial("#3c354a");

  switch (item.kind) {
    case "fridge": {
      const body = new THREE.Mesh(new THREE.BoxGeometry(0.95, 1.95, 0.82), mainMat);
      body.position.y = 0.98;
      body.castShadow = true;
      body.receiveShadow = true;
      group.add(body);

      const doorPivot = new THREE.Group();
      doorPivot.position.set(-0.47, 0.98, 0.42);
      doorPivot.rotation.y = item.active ? -0.58 : 0;
      const door = new THREE.Mesh(new THREE.BoxGeometry(0.92, 1.88, 0.08), mainMat);
      door.position.set(0.46, 0, 0);
      door.castShadow = true;
      doorPivot.add(door);

      const handle = new THREE.Mesh(new THREE.CapsuleGeometry(0.035, 0.38, 6, 10), creamMat);
      handle.position.set(0.8, 0.25, 0.07);
      doorPivot.add(handle);

      const magnet = new THREE.Mesh(new THREE.SphereGeometry(0.08, 12, 10), accentMat);
      magnet.scale.set(1, 1, 0.4);
      magnet.position.set(0.32, 0.45, 0.06);
      doorPivot.add(magnet);

      group.add(doorPivot);
      break;
    }
    case "stove": {
      const base = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.95, 0.86), mainMat);
      base.position.y = 0.48;
      base.castShadow = true;
      base.receiveShadow = true;
      group.add(base);

      const ovenWin = new THREE.Mesh(
        new THREE.BoxGeometry(0.8, 0.42, 0.06),
        item.active ? glowMaterial("#ffbe55", 0.55) : darkMat,
      );
      ovenWin.position.set(0, 0.36, 0.42);
      group.add(ovenWin);

      for (const kx of [-0.36, -0.12, 0.12, 0.36]) {
        const knob = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.06, 12), accentMat);
        knob.rotation.x = Math.PI / 2;
        knob.position.set(kx, 0.78, 0.44);
        group.add(knob);
      }

      for (const bx of [-0.3, 0.3]) {
        const burner = new THREE.Mesh(
          new THREE.CylinderGeometry(0.2, 0.2, 0.04, 16),
          item.active ? glowMaterial("#ff6b6b", 0.65) : darkMat,
        );
        burner.position.set(bx, 0.97, 0);
        group.add(burner);
      }

      const pot = new THREE.Mesh(new THREE.CylinderGeometry(0.23, 0.2, 0.28, 16), accentMat);
      pot.position.set(0.3, 1.13, 0);
      pot.castShadow = true;
      group.add(pot);

      const lid = new THREE.Mesh(new THREE.SphereGeometry(0.22, 14, 10, 0, Math.PI * 2, 0, Math.PI / 2), creamMat);
      lid.position.set(0.3, 1.26, 0);
      group.add(lid);
      break;
    }
    case "table": {
      const topGeo = item.styleVariant === 1
        ? new THREE.BoxGeometry(1.75, 0.14, 1.1)
        : new THREE.CylinderGeometry(0.92, 0.92, 0.14, 24);
      const top = new THREE.Mesh(topGeo, mainMat);
      top.position.y = 0.78;
      top.castShadow = true;
      top.receiveShadow = true;
      group.add(top);

      const cloth = new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.62, 0.02, 20), accentMat);
      cloth.position.y = 0.86;
      group.add(cloth);

      const legGeo = new THREE.CylinderGeometry(0.07, 0.06, 0.76, 12);
      for (const [lx, lz] of [[-0.5, -0.35], [0.5, -0.35], [-0.5, 0.35], [0.5, 0.35]] as const) {
        const leg = new THREE.Mesh(legGeo, mainMat);
        leg.position.set(lx, 0.38, lz);
        leg.castShadow = true;
        group.add(leg);
      }
      break;
    }
    case "rug": {
      const outer = item.styleVariant === 2
        ? new THREE.CylinderGeometry(1.32, 1.32, 0.05, 6)
        : new THREE.CylinderGeometry(1.32, 1.32, 0.05, 28);
      const base = new THREE.Mesh(outer, mainMat);
      base.position.y = 0.03;
      base.receiveShadow = true;
      group.add(base);

      const inner = new THREE.Mesh(new THREE.CylinderGeometry(0.86, 0.86, 0.06, 24), accentMat);
      inner.position.y = 0.035;
      inner.receiveShadow = true;
      group.add(inner);

      if (item.styleVariant === 1) {
        const dot = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.38, 0.07, 18), creamMat);
        dot.position.y = 0.04;
        group.add(dot);
      }
      break;
    }
    case "sofa": {
      const seat = new THREE.Mesh(new THREE.BoxGeometry(2.05, 0.45, 0.92), mainMat);
      seat.position.y = 0.34;
      seat.castShadow = true;
      seat.receiveShadow = true;
      group.add(seat);

      const back = new THREE.Mesh(new THREE.BoxGeometry(2.05, 0.78, 0.28), mainMat);
      back.position.set(0, 0.78, -0.34);
      back.castShadow = true;
      group.add(back);

      for (const sx of [-0.94, 0.94]) {
        const arm = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.56, 0.94), mainMat);
        arm.position.set(sx, 0.52, 0);
        arm.castShadow = true;
        group.add(arm);
      }

      const pillowXs = item.styleVariant === 1 ? [-0.55, 0, 0.55] : [-0.52, 0.52];
      for (const px of pillowXs) {
        const pillow = new THREE.Mesh(new THREE.SphereGeometry(0.22, 14, 12), accentMat);
        pillow.scale.set(1.15, 0.95, 0.55);
        pillow.position.set(px, 0.72, -0.1);
        pillow.castShadow = true;
        group.add(pillow);
      }
      break;
    }
    case "phoneTable": {
      const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.48, 0.72, 18), mainMat);
      stand.position.y = 0.36;
      stand.castShadow = true;
      stand.receiveShadow = true;
      group.add(stand);

      const phoneBody = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.2, 0.4), accentMat);
      phoneBody.position.set(0, 0.82, 0);
      phoneBody.castShadow = true;
      group.add(phoneBody);

      const handset = new THREE.Mesh(new THREE.CapsuleGeometry(0.08, 0.38, 8, 12), creamMat);
      handset.name = "phoneHandset";
      handset.rotation.z = Math.PI / 2;
      handset.position.set(0, 0.98, 0);
      handset.castShadow = true;
      group.add(handset);

      const dial = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.04, 14), glowMaterial("#ffe066", 0.5));
      dial.rotation.x = 0.35;
      dial.position.set(0, 0.9, 0.12);
      group.add(dial);
      break;
    }
    case "lamp": {
      const base = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.32, 0.12, 16), mainMat);
      base.position.y = 0.06;
      base.castShadow = true;
      group.add(base);

      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 1.4, 12), creamMat);
      pole.position.y = 0.76;
      pole.castShadow = true;
      group.add(pole);

      const shade = new THREE.Mesh(
        new THREE.ConeGeometry(0.42, 0.55, 18, 1, true),
        item.active ? glowMaterial(item.color, 0.45) : mainMat,
      );
      shade.material.side = THREE.DoubleSide;
      shade.position.y = 1.56;
      shade.castShadow = true;
      group.add(shade);

      if (item.active) {
        const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.11, 12, 10), glowMaterial("#fff3b0", 1.1));
        bulb.position.y = 1.42;
        group.add(bulb);
        const light = new THREE.PointLight("#ffe8a3", 2.2, 6.5, 1.5);
        light.position.set(0, 1.4, 0.25);
        group.add(light);
      }
      break;
    }
    case "bed": {
      const frame = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.34, 1.45), mainMat);
      frame.position.y = 0.22;
      frame.castShadow = true;
      frame.receiveShadow = true;
      group.add(frame);

      const headboard = new THREE.Mesh(new THREE.BoxGeometry(1.92, 0.96, 0.18), mainMat);
      headboard.position.set(0, 0.64, -0.66);
      headboard.castShadow = true;
      group.add(headboard);

      const mattress = new THREE.Mesh(new THREE.BoxGeometry(1.78, 0.26, 1.32), creamMat);
      mattress.position.y = 0.46;
      mattress.castShadow = true;
      group.add(mattress);

      const blanket = new THREE.Mesh(new THREE.BoxGeometry(1.82, 0.29, 0.88), accentMat);
      blanket.position.set(0, 0.48, 0.24);
      blanket.castShadow = true;
      group.add(blanket);

      const pillow = new THREE.Mesh(new THREE.SphereGeometry(0.24, 14, 12), creamMat);
      pillow.scale.set(1.9, 0.52, 0.9);
      pillow.position.set(0, 0.65, -0.36);
      pillow.castShadow = true;
      group.add(pillow);
      break;
    }
    case "toyChest": {
      const box = new THREE.Mesh(new THREE.BoxGeometry(1.05, 0.66, 0.72), mainMat);
      box.position.y = 0.34;
      box.castShadow = true;
      box.receiveShadow = true;
      group.add(box);

      const rim = new THREE.Mesh(new THREE.BoxGeometry(1.12, 0.12, 0.78), accentMat);
      rim.position.y = 0.7;
      rim.castShadow = true;
      group.add(rim);

      const blockColors = ["#ff6b8b", "#ffe066", "#74c0fc"] as const;
      blockColors.forEach((hex, idx) => {
        const block = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.22, 0.22), clayMaterial(hex, 0.5));
        block.position.set(-0.26 + idx * 0.26, 0.86, 0);
        block.rotation.y = idx * 0.35;
        block.castShadow = true;
        group.add(block);
      });
      break;
    }
    default: {
      const unreachable: never = item.kind;
      return unreachable;
    }
  }

  group.traverse((obj) => {
    obj.userData = { hitKind: "furniture", hitId: item.id };
  });

  return group;
}
