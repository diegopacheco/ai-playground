import * as THREE from "three";
import type { CharacterSpec } from "../../characters/types";
import { clayMaterial, glossyMaterial } from "./materials";

export function buildCharacterMesh(spec: CharacterSpec): THREE.Group {
  const root = new THREE.Group();
  root.userData = { hitKind: "character", hitId: spec.id };

  const bodyMat = clayMaterial(spec.bodyColor, 0.72);
  const bellyMat = clayMaterial(spec.bellyColor, 0.75);
  const accentMat = clayMaterial(spec.accentColor, 0.55);
  const eyeMat = glossyMaterial("#252333");
  const whiteMat = glossyMaterial("#ffffff");
  const blushMat = clayMaterial("#ff8fa3", 0.7);

  const shadowDisc = new THREE.Mesh(
    new THREE.CircleGeometry(0.42, 24),
    new THREE.MeshBasicMaterial({ color: 0x6b4f3a, transparent: true, opacity: 0.18 }),
  );
  shadowDisc.rotation.x = -Math.PI / 2;
  shadowDisc.position.y = 0.02;
  root.add(shadowDisc);

  const actor = new THREE.Group();
  actor.name = "actor";
  root.add(actor);

  const body = new THREE.Mesh(new THREE.SphereGeometry(0.36, 20, 18), bodyMat);
  body.scale.set(1.0, 1.08, 0.92);
  body.position.y = 0.42;
  body.castShadow = true;
  actor.add(body);

  const belly = new THREE.Mesh(new THREE.SphereGeometry(0.24, 16, 14), bellyMat);
  belly.scale.set(0.95, 1.05, 0.55);
  belly.position.set(0, 0.41, 0.23);
  actor.add(belly);

  const footGeo = new THREE.SphereGeometry(0.11, 14, 12);
  const leftFoot = new THREE.Mesh(footGeo, bodyMat);
  leftFoot.scale.set(1, 0.65, 1.25);
  leftFoot.position.set(-0.16, 0.08, 0.08);
  leftFoot.castShadow = true;
  actor.add(leftFoot);

  const rightFoot = new THREE.Mesh(footGeo, bodyMat);
  rightFoot.scale.set(1, 0.65, 1.25);
  rightFoot.position.set(0.16, 0.08, 0.08);
  rightFoot.castShadow = true;
  actor.add(rightFoot);

  const armGeo = new THREE.CapsuleGeometry(0.08, 0.16, 8, 12);
  const leftArm = new THREE.Mesh(armGeo, bodyMat);
  leftArm.name = "leftArm";
  leftArm.position.set(-0.37, 0.46, 0.04);
  leftArm.rotation.z = 0.45;
  leftArm.castShadow = true;
  actor.add(leftArm);

  const rightArm = new THREE.Mesh(armGeo, bodyMat);
  rightArm.name = "rightArm";
  rightArm.position.set(0.37, 0.46, 0.04);
  rightArm.rotation.z = -0.45;
  rightArm.castShadow = true;
  actor.add(rightArm);

  const head = new THREE.Group();
  head.name = "head";
  head.position.set(0, 0.96, 0.02);
  actor.add(head);

  const headMesh = new THREE.Mesh(new THREE.SphereGeometry(0.35, 22, 18), bodyMat);
  headMesh.scale.set(1.06, 0.96, 0.98);
  headMesh.castShadow = true;
  head.add(headMesh);

  const eyeY = spec.species === "frog" ? 0.14 : 0.04;
  const eyeOffsetX = 0.14;
  for (const side of [-1, 1] as const) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.062, 14, 12), eyeMat);
    eye.position.set(side * eyeOffsetX, eyeY, 0.3);
    head.add(eye);

    const sparkle = new THREE.Mesh(new THREE.SphereGeometry(0.022, 10, 8), whiteMat);
    sparkle.position.set(side * eyeOffsetX + 0.018, eyeY + 0.022, 0.348);
    head.add(sparkle);

    const cheek = new THREE.Mesh(new THREE.SphereGeometry(0.055, 12, 10), blushMat);
    cheek.scale.set(1.2, 0.72, 0.45);
    cheek.position.set(side * 0.22, -0.05, 0.27);
    head.add(cheek);
  }

  if (spec.species === "owl") {
    const beak = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.12, 12), accentMat);
    beak.rotation.x = Math.PI / 2;
    beak.position.set(0, -0.03, 0.35);
    head.add(beak);
  } else {
    const snout = new THREE.Mesh(new THREE.SphereGeometry(0.11, 14, 12), bellyMat);
    snout.scale.set(1.15, 0.82, 0.7);
    snout.position.set(0, -0.06, 0.29);
    head.add(snout);

    const nose = new THREE.Mesh(new THREE.SphereGeometry(0.042, 12, 10), eyeMat);
    nose.scale.set(1.25, 0.85, 0.8);
    nose.position.set(0, -0.02, 0.36);
    head.add(nose);
  }

  switch (spec.species) {
    case "bear": {
      for (const side of [-1, 1] as const) {
        const ear = new THREE.Mesh(new THREE.SphereGeometry(0.12, 14, 12), bodyMat);
        ear.position.set(side * 0.27, 0.27, -0.02);
        ear.castShadow = true;
        head.add(ear);
        const inner = new THREE.Mesh(new THREE.SphereGeometry(0.065, 12, 10), bellyMat);
        inner.position.set(side * 0.27, 0.27, 0.05);
        head.add(inner);
      }
      const bowtie = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.09, 0.07), accentMat);
      bowtie.position.set(0, 0.67, 0.28);
      actor.add(bowtie);
      break;
    }
    case "bunny": {
      for (const side of [-1, 1] as const) {
        const ear = new THREE.Mesh(new THREE.CapsuleGeometry(0.075, 0.26, 8, 12), bodyMat);
        ear.position.set(side * 0.14, 0.44, -0.02);
        ear.rotation.z = side * -0.16;
        ear.castShadow = true;
        head.add(ear);
        const inner = new THREE.Mesh(new THREE.CapsuleGeometry(0.04, 0.18, 6, 10), accentMat);
        inner.position.set(side * 0.14, 0.44, 0.04);
        inner.rotation.z = side * -0.16;
        head.add(inner);
      }
      break;
    }
    case "frog": {
      for (const side of [-1, 1] as const) {
        const bump = new THREE.Mesh(new THREE.SphereGeometry(0.11, 14, 12), bodyMat);
        bump.position.set(side * 0.18, 0.25, 0.14);
        head.add(bump);
      }
      const crown = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.16, 6), accentMat);
      crown.position.set(0, 0.38, 0.02);
      head.add(crown);
      break;
    }
    case "fox": {
      for (const side of [-1, 1] as const) {
        const ear = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.28, 12), bodyMat);
        ear.position.set(side * 0.22, 0.34, 0);
        ear.rotation.z = side * -0.25;
        ear.castShadow = true;
        head.add(ear);
      }
      const scarf = new THREE.Mesh(new THREE.TorusGeometry(0.24, 0.06, 10, 20), accentMat);
      scarf.rotation.x = Math.PI / 2;
      scarf.position.set(0, 0.69, 0.02);
      actor.add(scarf);
      break;
    }
    case "owl": {
      for (const side of [-1, 1] as const) {
        const tuft = new THREE.Mesh(new THREE.ConeGeometry(0.09, 0.18, 10), bodyMat);
        tuft.position.set(side * 0.22, 0.3, 0);
        tuft.rotation.z = side * -0.35;
        head.add(tuft);
        const ring = new THREE.Mesh(new THREE.TorusGeometry(0.095, 0.018, 8, 18), accentMat);
        ring.position.set(side * 0.14, 0.04, 0.31);
        head.add(ring);
      }
      break;
    }
    case "kitty": {
      for (const side of [-1, 1] as const) {
        const ear = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.22, 12), bodyMat);
        ear.position.set(side * 0.21, 0.32, 0);
        ear.rotation.z = side * -0.2;
        head.add(ear);
      }
      const bow = new THREE.Mesh(new THREE.SphereGeometry(0.08, 12, 10), accentMat);
      bow.scale.set(1.5, 0.8, 0.8);
      bow.position.set(0.17, 0.33, 0.16);
      head.add(bow);
      break;
    }
    default: {
      const unreachable: never = spec.species;
      return unreachable;
    }
  }

  root.traverse((obj) => {
    obj.userData = { hitKind: "character", hitId: spec.id };
  });

  return root;
}
