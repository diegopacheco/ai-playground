import {
  BoxGeometry,
  CapsuleGeometry,
  CircleGeometry,
  ConeGeometry,
  CylinderGeometry,
  DoubleSide,
  Group,
  Mesh,
  MeshStandardMaterial,
  PointLight,
  SphereGeometry,
  TorusGeometry,
} from "three";
import { linear, paint, toTexture } from "../art/canvas";

function mat(color: string, extra: Partial<{ roughness: number; metalness: number; emissive: string; emissiveIntensity: number }> = {}): MeshStandardMaterial {
  return new MeshStandardMaterial({ color, roughness: extra.roughness ?? 0.7, metalness: extra.metalness ?? 0, emissive: extra.emissive ?? "#000000", emissiveIntensity: extra.emissiveIntensity ?? 1 });
}

function put<T extends Mesh>(mesh: T, x: number, y: number, z: number): T {
  mesh.position.set(x, y, z);
  return mesh;
}

function barkTexture(): MeshStandardMaterial {
  const canvas = paint(256, 256, (ctx, w, h) => {
    ctx.fillStyle = linear(ctx, 0, h, ["#8b5a2b", "#6b4226", "#8b5a2b"]);
    ctx.fillRect(0, 0, w, h);
    for (let x = 0; x < w; x += 10) {
      ctx.fillStyle = x % 20 ? "rgba(40,20,5,0.35)" : "rgba(255,220,170,0.12)";
      ctx.fillRect(x, 0, 4, h);
    }
  });
  return new MeshStandardMaterial({ map: toTexture(canvas), roughness: 0.9 });
}

function ringTexture(): MeshStandardMaterial {
  const canvas = paint(256, 256, (ctx, w, h) => {
    ctx.fillStyle = "#f2c98a";
    ctx.fillRect(0, 0, w, h);
    for (let r = 12; r < 128; r += 14) {
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, r, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(140,90,40,0.6)";
      ctx.lineWidth = 3;
      ctx.stroke();
    }
  });
  return new MeshStandardMaterial({ map: toTexture(canvas), roughness: 0.8 });
}

export function workbench(): Group {
  const group = new Group();
  const wood = mat("#9a6332");
  group.add(put(new Mesh(new BoxGeometry(2.6, 0.16, 1.0), wood), 0, 0.9, 0));
  for (const [x, z] of [[-1.15, -0.4], [1.15, -0.4], [-1.15, 0.4], [1.15, 0.4]] as const) group.add(put(new Mesh(new BoxGeometry(0.14, 0.9, 0.14), wood), x, 0.45, z));
  group.add(put(new Mesh(new BoxGeometry(2.4, 0.08, 0.8), wood), 0, 0.25, 0));
  group.add(put(new Mesh(new BoxGeometry(0.3, 0.24, 0.3), mat("#555a66", { metalness: 0.6, roughness: 0.4 })), 1.05, 1.1, 0.25));
  const saw = put(new Mesh(new BoxGeometry(0.7, 0.18, 0.02), mat("#c9ccd6", { metalness: 0.8, roughness: 0.3 })), -0.7, 1.08, 0.3);
  saw.rotation.x = -Math.PI / 2;
  group.add(saw);
  return group;
}

export function log(length: number): Group {
  const group = new Group();
  const body = new Mesh(new CylinderGeometry(0.28, 0.3, length, 20), [barkTexture(), ringTexture(), ringTexture()]);
  body.rotation.z = Math.PI / 2;
  body.position.y = 0.3;
  group.add(body);
  return group;
}

export function coin(): Mesh {
  const mesh = new Mesh(new CylinderGeometry(0.13, 0.13, 0.03, 24), mat("#f2c14e", { metalness: 0.2, roughness: 0.35, emissive: "#c98a10", emissiveIntensity: 0.55 }));
  mesh.rotation.order = "YXZ";
  mesh.rotation.x = Math.PI / 2;
  return mesh;
}

export function miniTheater(): Group {
  const group = new Group();
  const gold = mat("#e9b949", { metalness: 0.5, roughness: 0.4 });
  const red = mat("#b51f2e", { roughness: 0.8 });
  group.add(put(new Mesh(new BoxGeometry(2.4, 0.5, 0.9), mat("#6b2a1a")), 0, 0.25, 0));
  for (const s of [-1, 1]) group.add(put(new Mesh(new BoxGeometry(0.22, 1.9, 0.3), gold), s * 1.1, 1.4, 0.3));
  group.add(put(new Mesh(new BoxGeometry(2.42, 0.36, 0.3), gold), 0, 2.3, 0.3));
  for (const s of [-1, 1]) {
    const drape = put(new Mesh(new CylinderGeometry(0.16, 0.24, 1.7, 12), red), s * 0.88, 1.3, 0.25);
    group.add(drape);
  }
  group.add(put(new Mesh(new BoxGeometry(2.2, 1.9, 0.04), mat("#2a1a4a")), 0, 1.4, -0.4));
  return group;
}

export function carousel(): { group: Group; spin: Group; horses: Mesh[] } {
  const group = new Group();
  const spin = new Group();
  group.add(spin);
  const roofCanvas = paint(512, 64, (ctx, w, h) => {
    for (let x = 0; x < w; x += 64) {
      ctx.fillStyle = (x / 64) % 2 ? "#ffffff" : "#e8505b";
      ctx.fillRect(x, 0, 64, h);
    }
  });
  spin.add(put(new Mesh(new CylinderGeometry(1.2, 1.25, 0.2, 32), mat("#ffd23f")), 0, 0.1, 0));
  spin.add(put(new Mesh(new CylinderGeometry(0.08, 0.08, 2.2, 12), mat("#e9b949", { metalness: 0.25 })), 0, 1.2, 0));
  spin.add(put(new Mesh(new ConeGeometry(1.4, 0.7, 32, 1), new MeshStandardMaterial({ map: toTexture(roofCanvas), roughness: 0.6 })), 0, 2.55, 0));
  spin.add(put(new Mesh(new SphereGeometry(0.12, 16, 12), mat("#ffd23f", { emissive: "#ffb52e", emissiveIntensity: 1.5 })), 0, 3.0, 0));
  const colors = ["#3fa8e8", "#ff6fa3", "#4caf50", "#ffd23f", "#9b6bff", "#ff8a3d"];
  const horses: Mesh[] = [];
  colors.forEach((color, i) => {
    const a = (i / colors.length) * Math.PI * 2;
    const x = Math.cos(a) * 0.9;
    const z = Math.sin(a) * 0.9;
    spin.add(put(new Mesh(new CylinderGeometry(0.03, 0.03, 2.1, 8), mat("#e9b949", { metalness: 0.25 })), x, 1.2, z));
    const horse = new Mesh(new CapsuleGeometry(0.14, 0.34, 6, 12), mat(color));
    horse.rotation.z = Math.PI / 2;
    horse.rotation.y = -a;
    const head = put(new Mesh(new SphereGeometry(0.13, 12, 10), mat(color)), 0.0, -0.28, 0.0);
    head.scale.set(1, 0.8, 1);
    horse.add(head);
    put(horse, x, 1.0, z);
    horses.push(horse);
    spin.add(horse);
  });
  return { group, spin, horses };
}

function jawHalf(top: boolean, skin: MeshStandardMaterial): Group {
  const half = new Group();
  const shell = new Mesh(new SphereGeometry(1, 32, 12, 0, Math.PI * 2, top ? 0 : Math.PI / 2, Math.PI / 2), skin);
  shell.material.side = DoubleSide;
  const inside = new Mesh(new CircleGeometry(1, 32), mat(top ? "#5a0a1a" : "#b8405a", { roughness: 0.9 }));
  inside.rotation.x = top ? Math.PI / 2 : -Math.PI / 2;
  inside.position.y = top ? -0.001 : 0.001;
  for (const part of [shell, inside]) {
    part.scale.set(1.7, top ? 0.85 : 0.6, 0.95);
    part.position.x = -1.3;
    half.add(part);
  }
  const tooth = new ConeGeometry(0.08, 0.26, 8);
  const white = mat("#ffffff", { roughness: 0.3 });
  for (let i = 0; i <= 10; i++) {
    const a = Math.PI / 2 + (i / 10) * Math.PI;
    const t = put(new Mesh(tooth, white), -1.3 + Math.cos(a) * 1.6, top ? -0.12 : 0.12, Math.sin(a) * 0.88);
    if (top) t.rotation.z = Math.PI;
    half.add(t);
  }
  return half;
}

export function shark(): { group: Group; upper: Group; lower: Group } {
  const group = new Group();
  const skin = mat("#7f9fc0", { roughness: 0.5 });
  const body = put(new Mesh(new SphereGeometry(1, 32, 20), skin), 1.5, 1.3, 0);
  body.scale.set(2.0, 1.0, 0.95);
  group.add(body);
  const tail = put(new Mesh(new ConeGeometry(0.7, 1.6, 24), skin), 3.9, 1.3, 0);
  tail.rotation.z = Math.PI / 2;
  group.add(tail);
  for (const s of [-1, 1]) {
    const fin = put(new Mesh(new ConeGeometry(0.5, 1.3, 4), skin), 4.8, 1.3 + s * 0.6, 0);
    fin.rotation.z = Math.PI / 2 + s * 0.6;
    fin.scale.z = 0.25;
    group.add(fin);
  }
  const dorsal = put(new Mesh(new ConeGeometry(0.45, 1.1, 4), skin), 1.6, 2.6, 0);
  dorsal.scale.z = 0.25;
  dorsal.rotation.z = -0.35;
  group.add(dorsal);
  const upper = jawHalf(true, skin);
  const lower = jawHalf(false, mat("#d9e3ea", { roughness: 0.6 }));
  for (const jaw of [upper, lower]) {
    jaw.position.set(0.5, 1.25, 0);
    group.add(jaw);
  }
  for (const z of [-0.62, 0.62]) {
    const eye = put(new Mesh(new SphereGeometry(0.16, 16, 12), mat("#ffffff")), -1.2, 0.5, z);
    eye.add(put(new Mesh(new SphereGeometry(0.08, 12, 10), mat("#111111")), -0.05, 0, z > 0 ? 0.1 : -0.1));
    upper.add(eye);
  }
  return { group, upper, lower };
}

export function candle(): { group: Group; flame: Mesh; light: PointLight } {
  const group = new Group();
  group.add(put(new Mesh(new CylinderGeometry(0.42, 0.42, 0.06, 24), mat("#9a6332")), 0, 0.62, 0));
  group.add(put(new Mesh(new CylinderGeometry(0.06, 0.06, 0.6, 8), mat("#9a6332")), 0, 0.3, 0));
  group.add(put(new Mesh(new CylinderGeometry(0.06, 0.07, 0.3, 16), mat("#fff6d8")), 0, 0.8, 0));
  const flame = put(new Mesh(new ConeGeometry(0.05, 0.16, 12), mat("#ffd27a", { emissive: "#ffb52e", emissiveIntensity: 4 })), 0, 1.03, 0);
  group.add(flame);
  const light = new PointLight("#ffb052", 6, 5, 1.6);
  light.position.set(0, 1.15, 0.1);
  group.add(light);
  return { group, flame, light };
}

export function ribs(count: number, radius: number): Group {
  const group = new Group();
  const bone = mat("#f2c6cf", { roughness: 0.8 });
  for (let i = 0; i < count; i++) {
    const rib = new Mesh(new TorusGeometry(radius, 0.09, 8, 40, Math.PI), bone);
    rib.position.z = -1.6 + i * (3.2 / Math.max(count - 1, 1));
    group.add(rib);
  }
  return group;
}

export function seaweed(height: number, color: string): { group: Group; joints: Group[] } {
  const group = new Group();
  const joints: Group[] = [];
  const material = mat(color, { roughness: 0.8 });
  let parent: Group = group;
  const seg = height / 6;
  for (let i = 0; i < 6; i++) {
    const joint = new Group();
    joint.position.y = i === 0 ? 0 : seg;
    const blade = put(new Mesh(new CapsuleGeometry(0.07 * (1 - i * 0.1), seg, 4, 8), material), 0, seg / 2, 0);
    joint.add(blade);
    parent.add(joint);
    joints.push(joint);
    parent = joint;
  }
  return { group, joints };
}

export function balloon(color: string): Group {
  const group = new Group();
  const skin = new Mesh(new SphereGeometry(0.22, 20, 16), new MeshStandardMaterial({ color, roughness: 0.25, metalness: 0.1, side: DoubleSide }));
  skin.scale.y = 1.2;
  skin.position.y = 1.4;
  group.add(skin);
  group.add(put(new Mesh(new CylinderGeometry(0.006, 0.006, 1.2, 4), mat("#ffffff")), 0, 0.6, 0));
  return group;
}

export function drum(): Group {
  const group = new Group();
  const stripes = paint(512, 128, (ctx, w, h) => {
    ctx.fillStyle = "#e8505b";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#ffd23f";
    ctx.lineWidth = 8;
    ctx.beginPath();
    for (let x = 0; x < w; x += 64) {
      ctx.moveTo(x, 10);
      ctx.lineTo(x + 32, h - 10);
      ctx.lineTo(x + 64, 10);
    }
    ctx.stroke();
  });
  group.add(put(new Mesh(new CylinderGeometry(0.35, 0.35, 0.45, 24), [new MeshStandardMaterial({ map: toTexture(stripes) }), mat("#fff6d8"), mat("#fff6d8")]), 0, 0.23, 0));
  return group;
}
