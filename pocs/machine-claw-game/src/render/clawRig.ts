import * as THREE from "three";
import { CABINET, CLAW, FINGER_SEGMENTS, FINGER_TIP } from "../game/layout";
import type { Arena } from "../physics/arena";

export type ClawRig = {
  readonly group: THREE.Group;
  sync(arena: Arena): void;
};

export function createClawRig(): ClawRig {
  const group = new THREE.Group();
  const chrome = new THREE.MeshStandardMaterial({ color: 0xf2f5fa, roughness: 0.14, metalness: 1 });
  const paint = new THREE.MeshStandardMaterial({ color: 0xffc928, roughness: 0.35, metalness: 0.4 });
  const rubber = new THREE.MeshStandardMaterial({ color: 0x1b1b22, roughness: 0.7 });

  const hub = new THREE.Group();
  const drum = new THREE.Mesh(new THREE.CylinderGeometry(CLAW.hubRadius, CLAW.hubRadius, CLAW.hubHalfHeight * 2, 28), paint);
  const cap = new THREE.Mesh(new THREE.ConeGeometry(CLAW.hubRadius * 0.8, 0.07, 28), chrome);
  cap.position.y = CLAW.hubHalfHeight + 0.035;
  drum.castShadow = true;
  hub.add(drum, cap);
  group.add(hub);

  const fingers = Array.from({ length: CLAW.fingerCount }, () => {
    const finger = new THREE.Group();
    for (const part of FINGER_SEGMENTS) {
      const bone = new THREE.Mesh(new THREE.CapsuleGeometry(CLAW.fingerRadius, part.halfLength * 2, 6, 12), chrome);
      bone.position.set(part.midX, part.midY, 0);
      bone.rotation.z = part.angle;
      bone.castShadow = true;
      finger.add(bone);
    }
    const tip = new THREE.Mesh(new THREE.SphereGeometry(CLAW.fingerRadius * 1.6, 12, 10), rubber);
    tip.position.set(FINGER_TIP.x, FINGER_TIP.y, 0);
    finger.add(tip);
    group.add(finger);
    return finger;
  });

  const cable = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 1, 8), rubber);
  const carriage = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.07, 0.14), paint);
  const rail = new THREE.Mesh(new THREE.BoxGeometry(CABINET.half * 2 - 0.06, 0.035, 0.05), chrome);
  group.add(cable, carriage, rail);

  function sync(arena: Arena): void {
    const at = arena.hub.translation();
    hub.position.set(at.x, at.y, at.z);
    fingers.forEach((finger, index) => {
      const body = arena.fingers[index];
      if (!body) return;
      const where = body.translation();
      const turn = body.rotation();
      finger.position.set(where.x, where.y, where.z);
      finger.quaternion.set(turn.x, turn.y, turn.z, turn.w);
    });
    const top = CABINET.railY - 0.035;
    const bottom = at.y + CLAW.hubHalfHeight + 0.06;
    cable.scale.y = Math.max(0.001, top - bottom);
    cable.position.set(at.x, (top + bottom) / 2, at.z);
    carriage.position.set(at.x, CABINET.railY, at.z);
    rail.position.set(0, CABINET.railY + 0.02, at.z);
  }

  return { group, sync };
}
