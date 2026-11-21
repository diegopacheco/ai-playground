import RAPIER from "@dimforge/rapier3d-compat";
import {
  BARRIER_BOXES,
  CABINET,
  CHUTE_BOXES,
  CLAW,
  FINGER_SEGMENTS,
  FINGER_TIP,
  FLOOR_BOXES,
  WALL_BOXES,
} from "../game/layout";

export type ToySpec = {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly radius: number;
  readonly look: number;
  readonly spin: number;
};

export type Toy = {
  readonly id: number;
  readonly look: number;
  readonly radius: number;
  readonly body: RAPIER.RigidBody;
};

export type ClawPose = {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly grip: number;
};

export type Arena = {
  readonly hub: RAPIER.RigidBody;
  readonly fingers: readonly RAPIER.RigidBody[];
  toys(): readonly Toy[];
  addToy(spec: ToySpec): Toy;
  placeClaw(pose: ClawPose, snap: boolean): void;
  clearance(pose: ClawPose): number;
  step(dt: number): void;
  takeFallen(): readonly Toy[];
};

const TOY = 0x0001;
const GRIPPER = 0x0002;
const FIXED = 0x0004;
const EVERYTHING = 0xffff;

function groups(member: number, filter: number): number {
  return ((member << 16) | filter) >>> 0;
}

export async function initPhysics(): Promise<void> {
  await RAPIER.init();
}

export function createArena(): Arena {
  const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
  const cabinet = world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
  for (const box of [...FLOOR_BOXES, ...BARRIER_BOXES, ...CHUTE_BOXES, ...WALL_BOXES]) {
    world.createCollider(
      RAPIER.ColliderDesc.cuboid(box.hx, box.hy, box.hz)
        .setTranslation(box.cx, box.cy, box.cz)
        .setFriction(0.9)
        .setCollisionGroups(groups(FIXED, TOY)),
      cabinet,
    );
  }

  const gripperGroups = groups(GRIPPER, TOY);
  const hub = world.createRigidBody(RAPIER.RigidBodyDesc.kinematicPositionBased());
  world.createCollider(
    RAPIER.ColliderDesc.cylinder(CLAW.hubHalfHeight, CLAW.hubRadius).setCollisionGroups(gripperGroups),
    hub,
  );
  const fingers = Array.from({ length: CLAW.fingerCount }, () => {
    const body = world.createRigidBody(RAPIER.RigidBodyDesc.kinematicPositionBased());
    for (const part of FINGER_SEGMENTS) {
      world.createCollider(
        RAPIER.ColliderDesc.capsule(part.halfLength, CLAW.fingerRadius)
          .setTranslation(part.midX, part.midY, 0)
          .setRotation({ x: 0, y: 0, z: Math.sin(part.angle / 2), w: Math.cos(part.angle / 2) })
          .setFriction(2)
          .setCollisionGroups(gripperGroups),
        body,
      );
    }
    world.createCollider(
      RAPIER.ColliderDesc.ball(CLAW.fingerRadius * 1.6)
        .setTranslation(FINGER_TIP.x, FINGER_TIP.y, 0)
        .setFriction(2)
        .setCollisionGroups(gripperGroups),
      body,
    );
    return body;
  });

  let toys: readonly Toy[] = [];
  let nextId = 1;

  function addToy(spec: ToySpec): Toy {
    const body = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(spec.x, spec.y, spec.z)
        .setRotation({ x: 0, y: Math.sin(spec.spin / 2), z: 0, w: Math.cos(spec.spin / 2) })
        .setLinearDamping(0.4)
        .setAngularDamping(1.2),
    );
    const toyGroups = groups(TOY, EVERYTHING);
    world.createCollider(
      RAPIER.ColliderDesc.ball(spec.radius)
        .setFriction(1.3)
        .setRestitution(0.05)
        .setDensity(60)
        .setCollisionGroups(toyGroups),
      body,
    );
    const toy: Toy = { id: nextId, look: spec.look, radius: spec.radius, body };
    nextId += 1;
    toys = [...toys, toy];
    return toy;
  }

  function placeClaw(pose: ClawPose, snap: boolean): void {
    const angle = CLAW.openAngle * (1 - pose.grip);
    const hubAt = { x: pose.x, y: pose.y, z: pose.z };
    if (snap) hub.setTranslation(hubAt, true);
    else hub.setNextKinematicTranslation(hubAt);
    fingers.forEach((finger, index) => {
      const yaw = -Math.PI / 2 + (index * 2 * Math.PI) / CLAW.fingerCount;
      const at = {
        x: pose.x + Math.cos(yaw) * CLAW.hingeRadius,
        y: pose.y - CLAW.hubHalfHeight,
        z: pose.z + Math.sin(yaw) * CLAW.hingeRadius,
      };
      const sy = Math.sin(-yaw / 2);
      const cy = Math.cos(-yaw / 2);
      const sz = Math.sin(angle / 2);
      const cz = Math.cos(angle / 2);
      const turn = { x: sy * sz, y: sy * cz, z: cy * sz, w: cy * cz };
      if (snap) {
        finger.setTranslation(at, true);
        finger.setRotation(turn, true);
      } else {
        finger.setNextKinematicTranslation(at);
        finger.setNextKinematicRotation(turn);
      }
    });
  }

  function clearance(pose: ClawPose): number {
    const ray = new RAPIER.Ray({ x: pose.x, y: pose.y - CLAW.hubHalfHeight, z: pose.z }, { x: 0, y: -1, z: 0 });
    const hit = world.castRay(ray, 3, true, undefined, groups(EVERYTHING, TOY | FIXED));
    return hit ? hit.timeOfImpact : Number.POSITIVE_INFINITY;
  }

  function step(dt: number): void {
    world.timestep = dt;
    world.step();
  }

  function takeFallen(): readonly Toy[] {
    const fallen = toys.filter((toy) => toy.body.translation().y < CABINET.fallLine);
    if (fallen.length === 0) return fallen;
    toys = toys.filter((toy) => !fallen.includes(toy));
    for (const toy of fallen) world.removeRigidBody(toy.body);
    return fallen;
  }

  return { hub, fingers, toys: () => toys, addToy, placeClaw, clearance, step, takeFallen };
}
