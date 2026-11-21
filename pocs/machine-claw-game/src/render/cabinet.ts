import * as THREE from "three";
import { BARRIER_BOXES, CABINET, CHUTE_BOXES, FLOOR_BOXES, type Box } from "../game/layout";

export type Cabinet = {
  readonly group: THREE.Group;
  readonly slot: THREE.Vector3;
  animate(time: number, moveX: number, moveZ: number, armed: boolean): void;
};

const OUTER = CABINET.half + CABINET.wallThickness;
const WIDTH = OUTER * 2;

function mesh(
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  x: number,
  y: number,
  z: number,
): THREE.Mesh {
  const made = new THREE.Mesh(geometry, material);
  made.position.set(x, y, z);
  return made;
}

function slab(material: THREE.Material, w: number, h: number, d: number, x: number, y: number, z: number): THREE.Mesh {
  return mesh(new THREE.BoxGeometry(w, h, d), material, x, y, z);
}

function fromBox(box: Box, material: THREE.Material): THREE.Mesh {
  return slab(material, box.hx * 2, box.hy * 2, box.hz * 2, box.cx, box.cy, box.cz);
}

function marquee(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 256;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("2d canvas is not available");
  const gradient = context.createLinearGradient(0, 0, 1024, 256);
  gradient.addColorStop(0, "#ff2d95");
  gradient.addColorStop(0.5, "#7a2dff");
  gradient.addColorStop(1, "#19d3ff");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 1024, 256);
  context.font = "900 150px 'Arial Black', Impact, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.shadowColor = "#fff36b";
  context.shadowBlur = 40;
  context.fillStyle = "#fffbe0";
  context.fillText("LUCKY CLAW", 512, 136);
  context.lineWidth = 5;
  context.strokeStyle = "#2a0845";
  context.shadowBlur = 0;
  context.strokeText("LUCKY CLAW", 512, 136);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

function room(): THREE.Group {
  const group = new THREE.Group();
  const ground = mesh(
    new THREE.PlaneGeometry(30, 30),
    new THREE.MeshStandardMaterial({ color: 0x1a1426, roughness: 0.28, metalness: 0.65 }),
    0,
    -CABINET.baseDepth,
    0,
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  group.add(ground);
  const wall = mesh(
    new THREE.PlaneGeometry(30, 12),
    new THREE.MeshStandardMaterial({ color: 0x150f24, roughness: 0.9 }),
    0,
    4,
    -4.5,
  );
  group.add(wall);
  const neon: readonly (readonly [number, number, number, number])[] = [
    [0xff2d95, -3.2, 1.9, 2.6],
    [0x19d3ff, 3.1, 2.4, 2.2],
    [0xfff36b, -1.2, 3.3, 1.4],
    [0x7a2dff, 1.6, 0.9, 1.8],
  ];
  for (const [color, x, y, length] of neon) {
    const bar = slab(new THREE.MeshBasicMaterial({ color, toneMapped: false }), length, 0.05, 0.05, x, y, -4.4);
    bar.rotation.z = (x * y) % 0.6;
    group.add(bar);
    const glow = new THREE.PointLight(color, 5, 9, 2);
    glow.position.set(x, y, -3.8);
    group.add(glow);
  }
  return group;
}

export function buildCabinet(): Cabinet {
  const group = new THREE.Group();
  group.add(room());

  const shell = new THREE.MeshStandardMaterial({ color: 0xf0206e, roughness: 0.32, metalness: 0.15 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x120c1c, roughness: 0.8 });
  const chrome = new THREE.MeshStandardMaterial({ color: 0xe9eef5, roughness: 0.16, metalness: 1 });
  const felt = new THREE.MeshStandardMaterial({ color: 0x0f8f9c, roughness: 0.95 });
  const acrylic = new THREE.MeshPhysicalMaterial({
    color: 0x9af4ff,
    transparent: true,
    opacity: 0.28,
    roughness: 0.05,
    depthWrite: false,
  });
  const glass = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.07,
    roughness: 0,
    envMapIntensity: 2.2,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  const baseY = -CABINET.baseDepth / 2 - CABINET.floorThickness / 2;
  const baseH = CABINET.baseDepth - CABINET.floorThickness;
  const t = CABINET.wallThickness;
  for (const side of [-1, 1]) {
    group.add(slab(shell, t, baseH, WIDTH, side * (OUTER - t / 2), baseY, 0));
    group.add(slab(shell, WIDTH, baseH, t, 0, baseY, side * (OUTER - t / 2)));
  }
  for (const box of FLOOR_BOXES) {
    const floor = fromBox(box, felt);
    floor.receiveShadow = true;
    group.add(floor);
  }
  for (const box of CHUTE_BOXES) group.add(fromBox(box, dark));
  for (const box of BARRIER_BOXES) group.add(fromBox(box, acrylic));
  group.add(slab(dark, WIDTH - 0.1, 0.02, WIDTH - 0.1, 0, -CABINET.baseDepth + 0.02, 0));

  const back = slab(
    new THREE.MeshStandardMaterial({ color: 0xbfa9ff, roughness: 0.12, metalness: 1 }),
    CABINET.half * 2,
    CABINET.wallHeight,
    0.02,
    0,
    CABINET.wallHeight / 2,
    -CABINET.half - 0.01,
  );
  group.add(back);
  const pane = new THREE.PlaneGeometry(CABINET.half * 2, CABINET.wallHeight);
  const front = mesh(pane, glass, 0, CABINET.wallHeight / 2, CABINET.half);
  const left = mesh(pane, glass, -CABINET.half, CABINET.wallHeight / 2, 0);
  const right = mesh(pane, glass, CABINET.half, CABINET.wallHeight / 2, 0);
  left.rotation.y = Math.PI / 2;
  right.rotation.y = Math.PI / 2;
  group.add(front, left, right);

  const leds: THREE.MeshBasicMaterial[] = [];
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      const x = sx * (OUTER - t / 2);
      const z = sz * (OUTER - t / 2);
      group.add(slab(chrome, t, CABINET.wallHeight, t, x, CABINET.wallHeight / 2, z));
      if (sz === 1) {
        const led = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
        leds.push(led);
        group.add(slab(led, 0.014, CABINET.wallHeight - 0.08, 0.004, x, CABINET.wallHeight / 2, z + t / 2));
      }
    }
  }

  const headY = CABINET.wallHeight + 0.2;
  group.add(slab(shell, WIDTH, 0.4, WIDTH, 0, headY, 0));
  const sign = mesh(
    new THREE.PlaneGeometry(WIDTH - 0.12, 0.3),
    new THREE.MeshBasicMaterial({ map: marquee(), toneMapped: false }),
    0,
    headY,
    OUTER + 0.002,
  );
  group.add(sign);
  const ceiling = mesh(
    new THREE.PlaneGeometry(CABINET.half * 2, CABINET.half * 2),
    new THREE.MeshBasicMaterial({ color: 0xd9cfc0, toneMapped: false }),
    0,
    CABINET.wallHeight - 0.001,
    0,
  );
  ceiling.rotation.x = Math.PI / 2;
  group.add(ceiling);
  for (const side of [-1, 1]) {
    group.add(slab(chrome, 0.03, 0.03, CABINET.half * 2, side * (CABINET.half - 0.03), CABINET.railY, 0));
  }

  const deckZ = OUTER + 0.14;
  group.add(slab(shell, WIDTH, 0.1, 0.28, 0, -0.13, deckZ));
  const deckLed = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
  leds.push(deckLed);
  group.add(slab(deckLed, WIDTH, 0.012, 0.004, 0, -0.13, deckZ + 0.142));

  const joystick = new THREE.Group();
  joystick.position.set(-0.18, -0.08, deckZ);
  joystick.add(mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.12, 12), chrome, 0, 0.06, 0));
  joystick.add(
    mesh(
      new THREE.SphereGeometry(0.035, 24, 16),
      new THREE.MeshStandardMaterial({ color: 0xff2030, roughness: 0.2 }),
      0,
      0.13,
      0,
    ),
  );
  group.add(joystick);
  group.add(mesh(new THREE.CylinderGeometry(0.05, 0.055, 0.012, 24), dark, -0.18, -0.076, deckZ));

  const buttonGlow = new THREE.MeshStandardMaterial({ color: 0xff2030, emissive: 0xff2030, emissiveIntensity: 0.2, roughness: 0.3 });
  group.add(mesh(new THREE.CylinderGeometry(0.06, 0.065, 0.012, 24), chrome, 0.2, -0.076, deckZ));
  const button = mesh(new THREE.CylinderGeometry(0.048, 0.048, 0.03, 24), buttonGlow, 0.2, -0.06, deckZ);
  group.add(button);

  const slot = new THREE.Vector3(0.42, -0.36, OUTER + 0.012);
  group.add(slab(chrome, 0.12, 0.18, 0.02, slot.x, slot.y, OUTER + 0.004));
  group.add(slab(dark, 0.012, 0.09, 0.022, slot.x, slot.y, OUTER + 0.005));

  group.add(slab(chrome, 0.4, 0.3, 0.02, -0.41, -0.48, OUTER + 0.004));
  group.add(slab(dark, 0.34, 0.24, 0.022, -0.41, -0.48, OUTER + 0.006));

  const lamp = new THREE.SpotLight(0xfff1dc, 7, 4, 0.85, 0.5, 1.4);
  lamp.position.set(0, CABINET.wallHeight - 0.02, 0.15);
  lamp.target.position.set(0, 0, 0);
  lamp.castShadow = true;
  lamp.shadow.mapSize.set(1024, 1024);
  lamp.shadow.camera.near = 0.2;
  lamp.shadow.camera.far = 3;
  lamp.shadow.bias = -0.0015;
  group.add(lamp, lamp.target);
  group.add(new THREE.HemisphereLight(0xbfd4ff, 0x2a1840, 0.45));
  const key = new THREE.DirectionalLight(0xffffff, 0.55);
  key.position.set(1.5, 3, 4);
  group.add(key);

  function animate(time: number, moveX: number, moveZ: number, armed: boolean): void {
    leds.forEach((led, index) => led.color.setHSL((time * 0.12 + index * 0.33) % 1, 1, 0.6));
    joystick.rotation.set(moveZ * 0.4, 0, -moveX * 0.4);
    buttonGlow.emissiveIntensity = armed ? 0.9 + Math.sin(time * 9) * 0.6 : 0.15;
    button.position.y = armed ? -0.06 : -0.066;
  }

  return { group, slot, animate };
}
