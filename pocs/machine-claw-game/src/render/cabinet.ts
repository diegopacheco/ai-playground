import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { BARRIER_BOXES, CABINET, CHUTE_BOXES, FLOOR_BOXES, type Box } from "../game/layout";
import { fitText, paintTexture } from "./canvasTexture";

export type Cabinet = {
  readonly group: THREE.Group;
  readonly slot: THREE.Vector3;
  animate(time: number, moveX: number, moveZ: number, armed: boolean): void;
};

const T = CABINET.wallThickness;
const OUTER = CABINET.half + T;
const WIDTH = OUTER * 2;
const INNER = CABINET.half * 2;
const FONT = "'Arial Black', Impact, sans-serif";

function at<M extends THREE.Object3D>(made: M, x: number, y: number, z: number): M {
  made.position.set(x, y, z);
  return made;
}

function slab(material: THREE.Material, w: number, h: number, d: number): THREE.Mesh {
  return new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
}

function soft(material: THREE.Material, w: number, h: number, d: number, radius: number): THREE.Mesh {
  return new THREE.Mesh(new RoundedBoxGeometry(w, h, d, 5, radius), material);
}

function fromBox(box: Box, material: THREE.Material): THREE.Mesh {
  return at(slab(material, box.hx * 2, box.hy * 2, box.hz * 2), box.cx, box.cy, box.cz);
}

function marquee(): THREE.CanvasTexture {
  return paintTexture(1024, 256, (context) => {
    const gradient = context.createLinearGradient(0, 0, 1024, 256);
    gradient.addColorStop(0, "#ff2d95");
    gradient.addColorStop(0.5, "#7a2dff");
    gradient.addColorStop(1, "#19d3ff");
    context.fillStyle = gradient;
    context.fillRect(0, 0, 1024, 256);
    context.fillStyle = "rgba(255,255,255,0.85)";
    for (let i = 0; i < 26; i += 1) {
      context.beginPath();
      context.arc(28 + i * 38.7, 22, 7, 0, Math.PI * 2);
      context.arc(28 + i * 38.7, 234, 7, 0, Math.PI * 2);
      context.fill();
    }
    context.textAlign = "center";
    context.textBaseline = "middle";
    fitText(context, "LUCKY CLAW", FONT, 150, 880);
    context.shadowColor = "#fff36b";
    context.shadowBlur = 36;
    context.fillStyle = "#fffbe0";
    context.fillText("LUCKY CLAW", 512, 132);
    context.shadowBlur = 0;
    context.lineWidth = 5;
    context.strokeStyle = "#2a0845";
    context.strokeText("LUCKY CLAW", 512, 132);
  });
}

function paint(): THREE.CanvasTexture {
  return paintTexture(64, 256, (context) => {
    const gradient = context.createLinearGradient(0, 0, 0, 256);
    gradient.addColorStop(0, "#ff4f9a");
    gradient.addColorStop(0.55, "#e0187a");
    gradient.addColorStop(1, "#7a1fd0");
    context.fillStyle = gradient;
    context.fillRect(0, 0, 64, 256);
  });
}

export function buildCabinet(): Cabinet {
  const group = new THREE.Group();
  const shell = new THREE.MeshPhysicalMaterial({ map: paint(), roughness: 0.35, clearcoat: 0.9, clearcoatRoughness: 0.15 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x15101f, roughness: 0.7 });
  const chrome = new THREE.MeshStandardMaterial({ color: 0xdfe6f0, roughness: 0.2, metalness: 1 });
  const lining = new THREE.MeshStandardMaterial({ color: 0x8c96ad, roughness: 0.42, metalness: 0.85 });
  const felt = new THREE.MeshStandardMaterial({ color: 0x0f8f9c, roughness: 0.95 });
  const acrylic = new THREE.MeshPhysicalMaterial({
    color: 0xbdf6ff,
    transparent: true,
    opacity: 0.2,
    roughness: 0.04,
    depthWrite: false,
  });
  const glass = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.06,
    roughness: 0,
    envMapIntensity: 2,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  const baseTop = -CABINET.floorThickness;
  const baseH = CABINET.baseDepth - CABINET.floorThickness;
  const baseY = baseTop - baseH / 2;
  for (const side of [-1, 1]) {
    group.add(at(slab(shell, WIDTH, baseH, T), 0, baseY, side * (OUTER - T / 2)));
    group.add(at(slab(shell, T, baseH, INNER), side * (OUTER - T / 2), baseY, 0));
    group.add(at(soft(chrome, WIDTH + 0.02, CABINET.floorThickness, T + 0.01, 0.012), 0, baseTop / 2, side * (OUTER - T / 2)));
    group.add(at(soft(chrome, T + 0.01, CABINET.floorThickness, INNER, 0.012), side * (OUTER - T / 2), baseTop / 2, 0));
  }
  group.add(at(soft(dark, WIDTH + 0.02, 0.07, WIDTH + 0.02, 0.015), 0, -CABINET.baseDepth + 0.035, 0));
  for (const box of FLOOR_BOXES) {
    const floor = fromBox(box, felt);
    floor.receiveShadow = true;
    group.add(floor);
  }
  for (const box of CHUTE_BOXES) group.add(fromBox(box, lining));
  for (const box of BARRIER_BOXES) {
    const wall = fromBox(box, acrylic);
    wall.renderOrder = 1;
    group.add(wall);
  }

  group.add(
    at(
      slab(new THREE.MeshStandardMaterial({ color: 0xb9a6ff, roughness: 0.1, metalness: 1 }), INNER, CABINET.wallHeight, 0.02),
      0,
      CABINET.wallHeight / 2,
      -CABINET.half - 0.01,
    ),
  );
  const pane = new THREE.PlaneGeometry(INNER, CABINET.wallHeight);
  const front = at(new THREE.Mesh(pane, glass), 0, CABINET.wallHeight / 2, CABINET.half);
  const left = at(new THREE.Mesh(pane, glass), -CABINET.half, CABINET.wallHeight / 2, 0);
  const right = at(new THREE.Mesh(pane, glass), CABINET.half, CABINET.wallHeight / 2, 0);
  left.rotation.y = Math.PI / 2;
  right.rotation.y = Math.PI / 2;
  for (const sheet of [front, left, right]) {
    sheet.renderOrder = 2;
    group.add(sheet);
  }

  const leds: THREE.MeshBasicMaterial[] = [];
  function led(w: number, h: number, d: number, x: number, y: number, z: number): void {
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
    leds.push(material);
    group.add(at(soft(material, w, h, d, 0.004), x, y, z));
  }

  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      const x = sx * (OUTER - T / 2);
      const z = sz * (OUTER - T / 2);
      group.add(at(soft(chrome, T, CABINET.wallHeight, T, 0.012), x, CABINET.wallHeight / 2, z));
      if (sz === 1) led(0.02, CABINET.wallHeight - 0.1, 0.01, x, CABINET.wallHeight / 2, z + T / 2);
    }
  }

  const headH = 0.42;
  const headY = CABINET.wallHeight + headH / 2;
  const headHalf = OUTER + 0.02;
  group.add(at(soft(shell, headHalf * 2, headH, headHalf * 2, 0.03), 0, headY, 0));
  group.add(at(soft(chrome, WIDTH - 0.08, 0.34, 0.02, 0.008), 0, headY, headHalf));
  group.add(
    at(
      new THREE.Mesh(new THREE.PlaneGeometry(WIDTH - 0.12, 0.3), new THREE.MeshBasicMaterial({ map: marquee(), toneMapped: false })),
      0,
      headY,
      headHalf + 0.012,
    ),
  );
  led(WIDTH - 0.06, 0.02, 0.01, 0, CABINET.wallHeight + 0.02, headHalf);
  const ceiling = at(new THREE.Mesh(new THREE.PlaneGeometry(INNER, INNER), dark), 0, CABINET.wallHeight - 0.002, 0);
  ceiling.rotation.x = Math.PI / 2;
  const panel = at(
    new THREE.Mesh(new THREE.PlaneGeometry(0.86, 0.86), new THREE.MeshBasicMaterial({ color: 0xfff1d6, toneMapped: false })),
    0,
    CABINET.wallHeight - 0.006,
    0,
  );
  panel.rotation.x = Math.PI / 2;
  group.add(ceiling, panel);
  for (const side of [-1, 1]) {
    group.add(at(soft(chrome, 0.03, 0.03, INNER - 0.01, 0.008), side * (CABINET.half - 0.03), CABINET.railY, 0));
  }

  const deckZ = OUTER + 0.14;
  group.add(at(soft(shell, WIDTH + 0.02, 0.1, 0.3, 0.03), 0, -0.13, deckZ));
  led(WIDTH - 0.06, 0.016, 0.01, 0, -0.13, deckZ + 0.15);

  const joystick = new THREE.Group();
  joystick.position.set(-0.18, -0.08, deckZ);
  joystick.add(at(new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.12, 16), chrome), 0, 0.06, 0));
  joystick.add(
    at(
      new THREE.Mesh(
        new THREE.SphereGeometry(0.035, 32, 20),
        new THREE.MeshPhysicalMaterial({ color: 0xff2030, roughness: 0.2, clearcoat: 1 }),
      ),
      0,
      0.13,
      0,
    ),
  );
  group.add(joystick);
  group.add(at(new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.055, 0.012, 32), dark), -0.18, -0.076, deckZ));

  const buttonGlow = new THREE.MeshStandardMaterial({ color: 0xff2030, emissive: 0xff2030, emissiveIntensity: 0.2, roughness: 0.3 });
  group.add(at(new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.065, 0.012, 32), chrome), 0.2, -0.076, deckZ));
  const button = at(new THREE.Mesh(new THREE.CylinderGeometry(0.048, 0.048, 0.03, 32), buttonGlow), 0.2, -0.06, deckZ);
  group.add(button);

  const slot = new THREE.Vector3(0.42, -0.42, OUTER + 0.016);
  group.add(at(soft(chrome, 0.13, 0.19, 0.02, 0.008), slot.x, slot.y, OUTER + 0.004));
  group.add(at(slab(dark, 0.012, 0.09, 0.006), slot.x, slot.y, OUTER + 0.013));

  group.add(at(soft(chrome, 0.42, 0.3, 0.02, 0.008), -0.38, -0.5, OUTER + 0.004));
  group.add(at(slab(dark, 0.36, 0.24, 0.006), -0.38, -0.5, OUTER + 0.013));

  const lamp = new THREE.SpotLight(0xfff1dc, 7, 4, 0.85, 0.5, 1.4);
  lamp.position.set(0, CABINET.wallHeight - 0.02, 0.15);
  lamp.target.position.set(0, 0, 0);
  lamp.castShadow = true;
  lamp.shadow.mapSize.set(2048, 2048);
  lamp.shadow.camera.near = 0.2;
  lamp.shadow.camera.far = 3;
  lamp.shadow.bias = -0.0008;
  lamp.shadow.normalBias = 0.01;
  lamp.shadow.radius = 4;
  group.add(lamp, lamp.target);
  group.add(new THREE.HemisphereLight(0xbfd4ff, 0x2a1840, 0.5));
  const key = new THREE.DirectionalLight(0xffffff, 0.6);
  key.position.set(1.5, 3, 4);
  group.add(key);

  function animate(time: number, moveX: number, moveZ: number, armed: boolean): void {
    leds.forEach((material, index) => material.color.setHSL((time * 0.12 + index * 0.21) % 1, 1, 0.6));
    joystick.rotation.set(moveZ * 0.4, 0, -moveX * 0.4);
    buttonGlow.emissiveIntensity = armed ? 0.9 + Math.sin(time * 9) * 0.6 : 0.15;
    button.position.y = armed ? -0.06 : -0.066;
  }

  return { group, slot, animate };
}
