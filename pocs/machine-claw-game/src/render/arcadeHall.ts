import * as THREE from "three";
import { CABINET } from "../game/layout";
import { fitText, paintTexture } from "./canvasTexture";

const FLOOR = -CABINET.baseDepth;
const NEON: readonly string[] = ["#ff2d95", "#19d3ff", "#fff36b", "#9a5bff", "#4dff9a", "#ff8a3d"];
const FONT = "'Arial Black', Impact, sans-serif";

function seeded(seed: number): () => number {
  let value = seed;
  return () => {
    value = (value * 1664525 + 1013904223) % 4294967296;
    return value / 4294967296;
  };
}

function pick(random: () => number): string {
  return NEON[Math.floor(random() * NEON.length)] ?? "#ffffff";
}

function carpet(): THREE.CanvasTexture {
  const texture = paintTexture(512, 512, (context) => {
    const random = seeded(11);
    context.fillStyle = "#15102e";
    context.fillRect(0, 0, 512, 512);
    context.lineCap = "round";
    for (let i = 0; i < 90; i += 1) {
      const x = 40 + random() * 432;
      const y = 40 + random() * 432;
      const size = 8 + random() * 22;
      const kind = Math.floor(random() * 4);
      context.strokeStyle = pick(random);
      context.fillStyle = context.strokeStyle;
      context.lineWidth = 3 + random() * 3;
      context.beginPath();
      if (kind === 0) {
        context.arc(x, y, size * 0.25, 0, Math.PI * 2);
        context.fill();
      } else if (kind === 1) {
        context.moveTo(x - size, y);
        context.bezierCurveTo(x - size / 2, y - size, x + size / 2, y + size, x + size, y);
        context.stroke();
      } else if (kind === 2) {
        context.moveTo(x, y - size * 0.7);
        context.lineTo(x + size * 0.7, y + size * 0.5);
        context.lineTo(x - size * 0.7, y + size * 0.5);
        context.closePath();
        context.stroke();
      } else {
        context.moveTo(x - size, y);
        context.lineTo(x - size / 3, y - size / 2);
        context.lineTo(x + size / 3, y + size / 2);
        context.lineTo(x + size, y);
        context.stroke();
      }
    }
  });
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(9, 9);
  return texture;
}

type ScreenPainter = (context: CanvasRenderingContext2D, random: () => number) => void;

const SCREENS: readonly ScreenPainter[] = [
  (context, random) => {
    context.fillStyle = "#05040f";
    context.fillRect(0, 0, 256, 192);
    for (let i = 0; i < 70; i += 1) {
      context.fillStyle = i % 5 === 0 ? pick(random) : "#ffffff";
      context.fillRect(random() * 256, random() * 192, 2, 2);
    }
    for (let row = 0; row < 3; row += 1) {
      for (let column = 0; column < 7; column += 1) {
        context.fillStyle = NEON[row] ?? "#ffffff";
        context.fillRect(34 + column * 28, 24 + row * 24, 16, 12);
      }
    }
    context.fillStyle = "#4dff9a";
    context.beginPath();
    context.moveTo(128, 150);
    context.lineTo(144, 174);
    context.lineTo(112, 174);
    context.fill();
  },
  (context) => {
    const sky = context.createLinearGradient(0, 0, 0, 100);
    sky.addColorStop(0, "#2b0a5c");
    sky.addColorStop(1, "#ff5fa2");
    context.fillStyle = sky;
    context.fillRect(0, 0, 256, 100);
    context.fillStyle = "#fff36b";
    context.beginPath();
    context.arc(128, 96, 42, Math.PI, 0);
    context.fill();
    context.fillStyle = "#0a0620";
    context.fillRect(0, 100, 256, 92);
    context.strokeStyle = "#19d3ff";
    context.lineWidth = 2;
    for (let i = -8; i <= 8; i += 1) {
      context.beginPath();
      context.moveTo(128 + i * 6, 100);
      context.lineTo(128 + i * 46, 192);
      context.stroke();
    }
    for (let i = 0; i < 6; i += 1) {
      const y = 100 + (i * i + 1) * 3;
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(256, y);
      context.stroke();
    }
  },
  (context) => {
    context.fillStyle = "#0d1b3a";
    context.fillRect(0, 0, 256, 192);
    context.fillStyle = "#3a3f55";
    context.beginPath();
    context.moveTo(108, 60);
    context.lineTo(148, 60);
    context.lineTo(250, 192);
    context.lineTo(6, 192);
    context.fill();
    context.fillStyle = "#fff36b";
    for (let i = 0; i < 5; i += 1) context.fillRect(124 - i, 68 + i * 26, 8 + i * 2, 14);
    context.fillStyle = "#ff2d95";
    context.fillRect(96, 132, 64, 30);
    context.fillStyle = "#19d3ff";
    context.fillRect(106, 122, 44, 14);
  },
  (context) => {
    context.fillStyle = "#07060f";
    context.fillRect(0, 0, 256, 192);
    context.textAlign = "center";
    context.fillStyle = "#fff36b";
    context.font = `900 30px ${FONT}`;
    context.fillText("HIGH SCORE", 128, 44);
    context.font = `900 18px ${FONT}`;
    ["AAA 99000", "ZED 87500", "KIT 64200", "MAX 51000"].forEach((line, index) => {
      context.fillStyle = NEON[index] ?? "#ffffff";
      context.fillText(line, 128, 82 + index * 26);
    });
  },
  (context, random) => {
    context.fillStyle = "#101226";
    context.fillRect(0, 0, 256, 192);
    for (let i = 0; i < 26; i += 1) {
      context.fillStyle = pick(random);
      context.beginPath();
      context.arc(20 + random() * 216, 20 + random() * 150, 5 + random() * 12, 0, Math.PI * 2);
      context.fill();
    }
  },
];

function neonSign(text: string, color: string, width: number): THREE.Mesh {
  const texture = paintTexture(1024, 256, (context) => {
    context.textAlign = "center";
    context.textBaseline = "middle";
    fitText(context, text, FONT, 170, 900);
    context.shadowColor = color;
    context.shadowBlur = 50;
    context.strokeStyle = color;
    context.lineWidth = 10;
    context.strokeText(text, 512, 132);
    context.shadowBlur = 18;
    context.strokeStyle = "#ffffff";
    context.lineWidth = 3;
    context.strokeText(text, 512, 132);
  });
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
    fog: false,
  });
  return new THREE.Mesh(new THREE.PlaneGeometry(width, width / 4), material);
}

function machines(): THREE.Group {
  const group = new THREE.Group();
  const random = seeded(5);
  const body = new THREE.BoxGeometry(0.86, 1.95, 0.8);
  const hood = new THREE.BoxGeometry(0.9, 0.3, 0.86);
  const deck = new THREE.BoxGeometry(0.86, 0.1, 0.36);
  const screen = new THREE.PlaneGeometry(0.66, 0.5);
  const screens = SCREENS.map(
    (paint) =>
      new THREE.MeshBasicMaterial({ map: paintTexture(256, 192, (context) => paint(context, random)), toneMapped: false }),
  );
  const paints = NEON.map(
    (color) => new THREE.MeshStandardMaterial({ color: new THREE.Color(color).multiplyScalar(0.22), roughness: 0.45 }),
  );
  const hoods = NEON.map((color) => new THREE.MeshBasicMaterial({ color, toneMapped: false }));

  function place(x: number, z: number, turn: number, index: number): void {
    const machine = new THREE.Group();
    const paint = paints[index % paints.length];
    const cabinet = new THREE.Mesh(body, paint);
    cabinet.position.y = 0.975;
    const top = new THREE.Mesh(hood, hoods[(index + 2) % hoods.length]);
    top.position.set(0, 2.05, 0.02);
    const panel = new THREE.Mesh(deck, paint);
    panel.position.set(0, 1.0, 0.5);
    const picture = new THREE.Mesh(screen, screens[index % screens.length]);
    picture.position.set(0, 1.45, 0.405);
    picture.rotation.x = -0.12;
    machine.add(cabinet, top, panel, picture);
    machine.position.set(x, FLOOR, z);
    machine.rotation.y = turn;
    group.add(machine);
  }

  for (let i = 0; i < 15; i += 1) place(-8.4 + i * 1.2, -7.2, 0, i);
  for (let i = 0; i < 9; i += 1) place(-9.2, -5.6 + i * 1.2, Math.PI / 2, i + 3);
  for (let i = 0; i < 4; i += 1) place(9.2, 1 + i * 1.2, -Math.PI / 2, i + 1);
  place(-4.6, -4.2, 0.5, 2);
  place(-5.7, -3.7, 0.5, 4);
  place(4.7, -4.4, -0.45, 0);
  place(5.8, -3.9, -0.45, 3);
  return group;
}

function prizeWall(): THREE.Group {
  const group = new THREE.Group();
  const wood = new THREE.MeshStandardMaterial({ color: 0x2a1c4a, roughness: 0.6 });
  const counter = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.05, 5.4), wood);
  counter.position.set(8.2, FLOOR + 0.525, -3.6);
  const glow = new THREE.Mesh(
    new THREE.BoxGeometry(0.02, 0.06, 5.4),
    new THREE.MeshBasicMaterial({ color: 0xfff36b, toneMapped: false }),
  );
  glow.position.set(7.59, FLOOR + 1.0, -3.6);
  group.add(counter, glow);
  const rows = 4;
  const perRow = 16;
  const prizes = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.15, 16, 12),
    new THREE.MeshStandardMaterial({ roughness: 0.9 }),
    rows * perRow,
  );
  const spot = new THREE.Object3D();
  const tint = new THREE.Color();
  const random = seeded(23);
  for (let row = 0; row < rows; row += 1) {
    const shelf = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.05, 5.4), wood);
    shelf.position.set(9.6, FLOOR + 1.5 + row * 0.62, -3.6);
    group.add(shelf);
    for (let column = 0; column < perRow; column += 1) {
      const index = row * perRow + column;
      spot.position.set(9.55, FLOOR + 1.68 + row * 0.62, -6.1 + column * 0.335);
      spot.updateMatrix();
      prizes.setMatrixAt(index, spot.matrix);
      prizes.setColorAt(index, tint.set(pick(random)).lerp(new THREE.Color(0xffffff), 0.25));
    }
  }
  group.add(prizes);
  return group;
}

function stringLights(): THREE.InstancedMesh {
  const strands = 6;
  const perStrand = 26;
  const bulbs = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.05, 8, 6),
    new THREE.MeshBasicMaterial({ toneMapped: false, fog: false }),
    strands * perStrand,
  );
  const spot = new THREE.Object3D();
  const tint = new THREE.Color();
  for (let strand = 0; strand < strands; strand += 1) {
    for (let bulb = 0; bulb < perStrand; bulb += 1) {
      const along = bulb / (perStrand - 1);
      const index = strand * perStrand + bulb;
      spot.position.set(-9.5 + along * 19, FLOOR + 4.9 - Math.sin(along * Math.PI) * 0.7, -6.5 + strand * 1.9);
      spot.updateMatrix();
      bulbs.setMatrixAt(index, spot.matrix);
      bulbs.setColorAt(index, tint.set(NEON[(bulb + strand) % NEON.length] ?? "#ffffff"));
    }
  }
  return bulbs;
}

export function buildArcadeHall(): THREE.Group {
  const group = new THREE.Group();
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(26, 26),
    new THREE.MeshStandardMaterial({ map: carpet(), color: 0x9a9a9a, roughness: 0.92 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = FLOOR;
  group.add(ground);

  const shade = new THREE.Mesh(
    new THREE.PlaneGeometry(3.4, 3.4),
    new THREE.MeshBasicMaterial({
      map: paintTexture(128, 128, (context) => {
        const fade = context.createRadialGradient(64, 64, 10, 64, 64, 64);
        fade.addColorStop(0, "rgba(0,0,0,0.75)");
        fade.addColorStop(1, "rgba(0,0,0,0)");
        context.fillStyle = fade;
        context.fillRect(0, 0, 128, 128);
      }),
      transparent: true,
      depthWrite: false,
    }),
  );
  shade.rotation.x = -Math.PI / 2;
  shade.position.set(0, FLOOR + 0.004, 0.1);
  group.add(shade);

  const wallPaint = new THREE.MeshStandardMaterial({ color: 0x1d1338, roughness: 0.85 });
  const stripe = new THREE.MeshBasicMaterial({ color: 0xff2d95, toneMapped: false });
  const stripeBlue = new THREE.MeshBasicMaterial({ color: 0x19d3ff, toneMapped: false });
  const walls: readonly (readonly [x: number, z: number, turn: number])[] = [
    [0, -8, 0],
    [-10, 0, Math.PI / 2],
    [10, 0, -Math.PI / 2],
    [0, 11, Math.PI],
  ];
  for (const [x, z, turn] of walls) {
    const wall = new THREE.Group();
    const face = new THREE.Mesh(new THREE.PlaneGeometry(26, 6.4), wallPaint);
    face.position.y = FLOOR + 3.2;
    const high = new THREE.Mesh(new THREE.PlaneGeometry(26, 0.06), stripe);
    high.position.set(0, FLOOR + 3.05, 0.01);
    const low = new THREE.Mesh(new THREE.PlaneGeometry(26, 0.06), stripeBlue);
    low.position.set(0, FLOOR + 2.85, 0.01);
    wall.add(face, high, low);
    wall.position.set(x, 0, z);
    wall.rotation.y = turn;
    group.add(wall);
  }
  const roof = new THREE.Mesh(new THREE.PlaneGeometry(26, 26), new THREE.MeshStandardMaterial({ color: 0x0c081a, roughness: 1 }));
  roof.rotation.x = Math.PI / 2;
  roof.position.y = FLOOR + 6.4;
  group.add(roof);

  const arcade = neonSign("ARCADE", "#ff2d95", 5);
  arcade.position.set(-4.6, FLOOR + 3.85, -7.95);
  const prizes = neonSign("PRIZES", "#fff36b", 4.4);
  prizes.position.set(9.95, FLOOR + 4.3, -3.6);
  prizes.rotation.y = -Math.PI / 2;
  const play = neonSign("PLAY TO WIN", "#19d3ff", 5);
  play.position.set(4.6, FLOOR + 3.85, -7.95);
  group.add(arcade, prizes, play, machines(), prizeWall(), stringLights());

  const lamps: readonly (readonly [color: number, x: number, y: number, z: number, power: number])[] = [
    [0xff2d95, -5.5, 2.6, -4.5, 34],
    [0x19d3ff, 5.5, 2.6, -4.5, 34],
    [0xffe2b0, 0, 3.6, 3.5, 26],
    [0x9a5bff, 0, 2.4, -6, 28],
  ];
  for (const [color, x, y, z, power] of lamps) {
    const lamp = new THREE.PointLight(color, power, 16, 2);
    lamp.position.set(x, FLOOR + y, z);
    group.add(lamp);
  }
  return group;
}
