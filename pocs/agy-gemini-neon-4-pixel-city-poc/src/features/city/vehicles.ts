import * as THREE from "three";
import { samplePolyline, VEHICLE_ROUTES } from "./catalog";
import { PALETTE } from "./palette";
import { createOutlinedBox } from "./geometry";
import type { VehicleRoute } from "./types";

type ActiveCar = {
  readonly mesh: THREE.Group;
  readonly route: VehicleRoute;
};

type ActiveWalker = {
  readonly mesh: THREE.Group;
  readonly start: readonly [number, number];
  readonly end: readonly [number, number];
  readonly speed: number;
  readonly phase: number;
};

export type TrafficRig = {
  readonly group: THREE.Group;
  readonly update: (elapsed: number) => void;
};

function computeBridgeElevation(x: number, z: number, baseY: number): number {
  if (Math.abs(x + 6.8) > 1.2) {
    return baseY;
  }
  if (z <= 1.6 && z >= -10.8) {
    return 1.38;
  }
  if (z > 1.6 && z < 3.4) {
    const t = 1 - (z - 1.6) / 1.8;
    return baseY + (1.38 - baseY) * t;
  }
  return baseY;
}

function createPixelCar(color: number): THREE.Group {
  const car = new THREE.Group();

  const lower = createOutlinedBox({
    width: 0.44,
    height: 0.22,
    depth: 0.78,
    color,
  });
  lower.position.y = 0.14;
  car.add(lower);

  const cabin = createOutlinedBox({
    width: 0.36,
    height: 0.18,
    depth: 0.42,
    color: PALETTE.ferryClock,
  });
  cabin.position.set(0, 0.32, -0.04);
  car.add(cabin);

  const wheelMat = new THREE.MeshLambertMaterial({ color: PALETTE.ink });
  for (const wx of [-0.22, 0.22]) {
    for (const wz of [-0.22, 0.22]) {
      const wheel = new THREE.Mesh(
        new THREE.BoxGeometry(0.08, 0.12, 0.14),
        wheelMat,
      );
      wheel.position.set(wx, 0.06, wz);
      car.add(wheel);
    }
  }

  return car;
}

function createCableCar(): THREE.Group {
  const trolley = new THREE.Group();

  const lower = createOutlinedBox({
    width: 1.5,
    height: 0.36,
    depth: 0.62,
    color: PALETTE.cableCarRed,
  });
  lower.position.y = 0.26;
  trolley.add(lower);

  const upper = createOutlinedBox({
    width: 1.28,
    height: 0.34,
    depth: 0.56,
    color: PALETTE.cableCarOchre,
  });
  upper.position.y = 0.6;
  trolley.add(upper);

  const roof = createOutlinedBox({
    width: 1.4,
    height: 0.12,
    depth: 0.62,
    color: PALETTE.ink,
  });
  roof.position.y = 0.82;
  trolley.add(roof);

  const lantern = createOutlinedBox({
    width: 0.85,
    height: 0.1,
    depth: 0.32,
    color: PALETTE.ferryClock,
  });
  lantern.position.y = 0.92;
  trolley.add(lantern);

  trolley.traverse((obj) => {
    obj.userData.landmarkId = "cable-car";
  });
  return trolley;
}

function createBayBoat(): THREE.Group {
  const boat = new THREE.Group();

  const hull = createOutlinedBox({
    width: 0.92,
    height: 0.22,
    depth: 0.44,
    color: PALETTE.bridgeRed,
  });
  hull.position.y = 0.11;
  boat.add(hull);

  const deck = createOutlinedBox({
    width: 0.6,
    height: 0.22,
    depth: 0.32,
    color: PALETTE.ferryClock,
  });
  deck.position.set(-0.04, 0.32, 0);
  boat.add(deck);

  const stack = createOutlinedBox({
    width: 0.14,
    height: 0.22,
    depth: 0.14,
    color: PALETTE.carYellow,
  });
  stack.position.set(0.08, 0.52, 0);
  boat.add(stack);

  return boat;
}

function createPedestrian(shirtColor: number): THREE.Group {
  const person = new THREE.Group();

  const legs = createOutlinedBox({
    width: 0.14,
    height: 0.18,
    depth: 0.12,
    color: PALETTE.ink,
    outline: false,
  });
  legs.position.y = 0.09;
  person.add(legs);

  const torso = createOutlinedBox({
    width: 0.18,
    height: 0.2,
    depth: 0.14,
    color: shirtColor,
    outline: false,
  });
  torso.position.y = 0.28;
  person.add(torso);

  const head = createOutlinedBox({
    width: 0.16,
    height: 0.16,
    depth: 0.16,
    color: 0xf0c9a5,
    outline: false,
  });
  head.position.y = 0.46;
  person.add(head);

  return person;
}

export function buildTraffic(): TrafficRig {
  const group = new THREE.Group();

  const cars: ActiveCar[] = VEHICLE_ROUTES.map((route) => {
    const mesh = createPixelCar(route.color);
    group.add(mesh);
    return { mesh, route };
  });

  const cableCar = createCableCar();
  group.add(cableCar);

  const boat = createBayBoat();
  group.add(boat);

  const walkerSpecs: readonly {
    readonly start: readonly [number, number];
    readonly end: readonly [number, number];
    readonly color: number;
    readonly speed: number;
    readonly phase: number;
  }[] = [
    { start: [-5.4, 9.6], end: [-5.4, 4.2], color: PALETTE.carRed, speed: 0.14, phase: 0.1 },
    { start: [-4.2, 5.8], end: [4.8, 5.8], color: PALETTE.carBlue, speed: 0.11, phase: 0.4 },
    { start: [1.4, 8.6], end: [8.2, 8.6], color: PALETTE.carYellow, speed: 0.13, phase: 0.7 },
    { start: [8.4, 7.6], end: [8.4, 0.2], color: PALETTE.awningGreen, speed: 0.12, phase: 0.25 },
    { start: [0.9, -3.6], end: [0.9, -8.2], color: PALETTE.victorianPink, speed: 0.15, phase: 0.6 },
    { start: [-3.2, 2.2], end: [-3.2, -1.6], color: PALETTE.ferryBlue, speed: 0.12, phase: 0.85 },
  ];

  const walkers: ActiveWalker[] = walkerSpecs.map((spec) => {
    const mesh = createPedestrian(spec.color);
    group.add(mesh);
    return {
      mesh,
      start: spec.start,
      end: spec.end,
      speed: spec.speed,
      phase: spec.phase,
    };
  });

  function update(elapsed: number): void {
    for (const car of cars) {
      const sample = samplePolyline(
        car.route.points,
        elapsed * car.route.speed + car.route.phase,
      );
      const y = computeBridgeElevation(sample.x, sample.z, car.route.elevation);
      car.mesh.position.set(sample.x, y, sample.z);
      car.mesh.rotation.y = sample.angle;
    }

    const pingPong = 0.5 - 0.5 * Math.cos(elapsed * 0.42);
    const cableX = -5.2 + pingPong * 12.2;
    cableCar.position.set(cableX, 0.08, 4.9);

    const boatPhase = elapsed * 0.28;
    const boatX = -5.6 + Math.sin(boatPhase) * 2.8;
    const boatZ = -7.6 + Math.cos(boatPhase * 0.7) * 1.4;
    boat.position.set(boatX, 0.05 + Math.sin(elapsed * 2.4) * 0.03, boatZ);
    boat.rotation.y = Math.atan2(
      Math.cos(boatPhase) * 2.8,
      -Math.sin(boatPhase * 0.7) * 0.98,
    );

    for (const walker of walkers) {
      const cycle = 0.5 - 0.5 * Math.cos(elapsed * walker.speed * Math.PI + walker.phase * 6.28);
      const wx = walker.start[0] + (walker.end[0] - walker.start[0]) * cycle;
      const wz = walker.start[1] + (walker.end[1] - walker.start[1]) * cycle;
      const hop = Math.abs(Math.sin(elapsed * 8 + walker.phase * 10)) * 0.04;
      walker.mesh.position.set(wx, 0.05 + hop, wz);
    }
  }

  update(0);
  return { group, update };
}
