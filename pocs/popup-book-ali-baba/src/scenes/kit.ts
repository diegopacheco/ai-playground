import {
  ConeGeometry,
  CylinderGeometry,
  DoubleSide,
  Group,
  IcosahedronGeometry,
  InstancedMesh,
  LatheGeometry,
  Matrix4,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  PlaneGeometry,
  type Texture,
  Vector2,
  Vector3,
} from "three";
import { arm, type ArmOptions, type FigureArt, man, type ManOptions, woman } from "../art/people";
import { latticeTexture } from "../art/textures";
import { Act, castShadow } from "../engine/act";
import { lampUniform, MASK_LAYER, maskMaterial } from "../engine/materials";
import { Sil } from "../engine/silhouette";

export interface Assets {
  spark: Texture;
  disc: Texture;
}

export type SceneBuilder = (assets: Assets) => Act;

export interface Puppet {
  body: Sil;
  arm: Sil;
  hand: [number, number];
}

export function joint(art: FigureArt, name: string): [number, number] {
  const [ux, uy] = art.joints[name] ?? [art.w / 2, 0];
  const k = art.worldH / art.h;
  return [(ux - art.w / 2) * k, (art.h - uy) * k];
}

export function sil(art: FigureArt): Sil {
  return new Sil(art.canvas, art.worldH);
}

function limb(art: FigureArt): Sil {
  const [px, py] = art.joints.pivot ?? [art.w / 2, 0];
  return new Sil(art.canvas, art.worldH, { anchorX: px / art.w, anchorY: 1 - py / art.h });
}

export function puppet(body: FigureArt, armOptions: Omit<ArmOptions, "worldScale">): Puppet {
  const b = sil(body);
  const armArt = arm({ ...armOptions, worldScale: body.worldH });
  const a = limb(armArt);
  const [x, y] = joint(body, "shoulder");
  b.attach(a, x, y);
  const pivot = armArt.joints.pivot ?? [0, 0];
  const hand = armArt.joints.hand ?? [0, 0];
  return { body: b, arm: a, hand: [0, -(hand[1] - pivot[1]) * (armArt.worldH / armArt.h)] };
}

const ALI: ManOptions = { turban: "round", beard: "short", build: "slim", seed: 3 };
const KASSIM: ManOptions = { turban: "big", beard: "long", build: "fat", mustache: true, worldH: 1.75, seed: 5 };
const CAPTAIN: ManOptions = { turban: "pointed", beard: "pointed", build: "broad", mustache: true, cape: true, sword: true, worldH: 1.92, seed: 7 };
const THIEF: ManOptions = { turban: "pointed", beard: "short", build: "slim", mustache: true, sword: true, worldH: 1.62, seed: 11 };
const MERCHANT: ManOptions = { turban: "tall", beard: "pointed", build: "broad", mustache: true, cape: true, worldH: 1.92, seed: 13 };
const SON: ManOptions = { turban: "tall", beard: "none", build: "slim", worldH: 1.72, seed: 15 };

export const cast = {
  ali: (hold: ArmOptions["hold"] = "none") => puppet(man(ALI), { hold }),
  kassim: () => puppet(man(KASSIM), {}),
  captain: (hold: ArmOptions["hold"] = "none") => puppet(man(CAPTAIN), { sleeve: "fitted", hold }),
  thief: (hold: ArmOptions["hold"] = "scimitar") => puppet(man(THIEF), { sleeve: "fitted", hold }),
  merchant: (hold: ArmOptions["hold"] = "none") => puppet(man(MERCHANT), { hold }),
  son: () => puppet(man(SON), {}),
  morgiana: (hold: ArmOptions["hold"] = "none") => puppet(woman(), { sleeve: "fitted", bangles: true, hold, length: 36 }),
};

export function walkBob(target: Sil, time: number, speed: number, height = 0.05): void {
  target.root.position.y = Math.abs(Math.sin(time * speed)) * height;
  target.root.rotation.z = Math.sin(time * speed) * 0.03;
}

export function flip(target: Sil): Sil {
  target.root.scale.x = -1;
  return target;
}

export function framesFor(count: number, draw: (frame: number) => HTMLCanvasElement): HTMLCanvasElement[] {
  return Array.from({ length: count }, (_, i) => draw(i));
}

const terracotta = new MeshStandardMaterial({ color: "#a4532c", roughness: 0.75 });
const brass = new MeshStandardMaterial({ color: "#c8902f", roughness: 0.35, metalness: 0.85 });
const rock = new MeshStandardMaterial({ color: "#6b5a4a", roughness: 0.95, flatShading: true });
const goldCoin = new MeshStandardMaterial({ color: "#e8b84a", roughness: 0.3, metalness: 0.95 });

const JAR_PROFILE = [
  [0, 0],
  [0.2, 0],
  [0.3, 0.12],
  [0.42, 0.42],
  [0.44, 0.62],
  [0.36, 0.86],
  [0.2, 0.98],
  [0.16, 1.06],
  [0.22, 1.12],
  [0.2, 1.16],
].map(([x, y]) => new Vector2(x, y));

export interface Jar {
  group: Group;
  lid: Group;
}

export function jar(scale = 1): Jar {
  const group = new Group();
  group.add(new Mesh(new LatheGeometry(JAR_PROFILE, 28), terracotta));
  const lid = new Group();
  const disc = new Mesh(new CylinderGeometry(0.24, 0.24, 0.05, 24), brass);
  const knob = new Mesh(new ConeGeometry(0.06, 0.1, 12), brass);
  knob.position.y = 0.07;
  lid.add(disc, knob);
  lid.position.y = 1.17;
  group.add(lid);
  group.scale.set(scale, scale * 1.3, scale);
  castShadow(group, 1, 0.5);
  return { group, lid };
}

export function lattice(radius = 0.38, height = 0.8): Group {
  const group = new Group();
  const texture = latticeTexture();
  texture.repeat.set(2, 1);
  const shell = new Mesh(new CylinderGeometry(radius, radius, height, 8, 1, true), new MeshStandardMaterial({ map: texture, alphaTest: 0.5, side: DoubleSide, metalness: 0.7, roughness: 0.35 }));
  const top = new Mesh(new ConeGeometry(radius * 1.1, height * 0.45, 8), brass);
  top.position.y = height * 0.72;
  const bottom = new Mesh(new ConeGeometry(radius * 1.05, height * 0.25, 8), brass);
  bottom.rotation.x = Math.PI;
  bottom.position.y = -height * 0.62;
  const chain = new Mesh(new CylinderGeometry(0.012, 0.012, 3, 6), brass);
  chain.position.y = height + 1.4;
  group.add(shell, top, bottom, chain);
  castShadow(group, 1, 0.22);
  return group;
}

export function boulder(radius = 0.95): Mesh {
  const geometry = new IcosahedronGeometry(radius, 2);
  const position = geometry.attributes.position!;
  const v = new Vector3();
  for (let i = 0; i < position.count; i++) {
    v.fromBufferAttribute(position, i);
    const n = 1 + Math.sin(v.x * 5.1) * 0.06 + Math.sin(v.y * 6.3 + v.z * 2) * 0.07 + Math.cos(v.z * 7.7) * 0.05;
    v.multiplyScalar(n);
    position.setXYZ(i, v.x, v.y * 1.08, v.z * 0.8);
  }
  geometry.computeVertexNormals();
  const mesh = new Mesh(geometry, rock);
  castShadow(mesh, 1, 0.5);
  return mesh;
}

interface Particle {
  p: Vector3;
  v: Vector3;
  spin: Vector3;
  rot: Vector3;
  life: number;
  max: number;
  size: number;
}

export class ShadowSpray {
  readonly mesh: InstancedMesh;
  readonly visible?: InstancedMesh;
  private readonly items: Particle[] = [];
  private readonly dummy = new Object3D();
  private next = 0;

  constructor(private readonly count: number, map: Texture | null, density: number, soft: number, private readonly gravity: number, solid?: MeshStandardMaterial) {
    const geometry = solid ? new CylinderGeometry(0.5, 0.5, 0.08, 14) : new PlaneGeometry(1, 1);
    this.mesh = new InstancedMesh(geometry, maskMaterial(map, density, soft), count);
    this.mesh.layers.set(MASK_LAYER);
    this.mesh.frustumCulled = false;
    if (solid) {
      this.visible = new InstancedMesh(geometry, solid, count);
      this.visible.frustumCulled = false;
      this.visible.instanceMatrix = this.mesh.instanceMatrix;
    }
    for (let i = 0; i < count; i++) this.items.push({ p: new Vector3(), v: new Vector3(), spin: new Vector3(), rot: new Vector3(), life: 0, max: 1, size: 0 });
    this.hideAll();
  }

  addTo(act: Act): this {
    act.free(this.mesh);
    if (this.visible) act.free(this.visible);
    return this;
  }

  emit(p: Vector3, v: Vector3, life: number, size: number, spin = 0): void {
    const item = this.items[this.next];
    this.next = (this.next + 1) % this.count;
    if (!item) return;
    item.p.copy(p);
    item.v.copy(v);
    item.life = life;
    item.max = life;
    item.size = size;
    item.rot.set(Math.random() * 6, Math.random() * 6, Math.random() * 6);
    item.spin.set((Math.random() - 0.5) * spin, (Math.random() - 0.5) * spin, (Math.random() - 0.5) * spin);
  }

  update(dt: number, grow = 0): void {
    const m = new Matrix4();
    for (let i = 0; i < this.count; i++) {
      const item = this.items[i];
      if (!item) continue;
      if (item.life <= 0) {
        this.mesh.setMatrixAt(i, m.makeScale(0, 0, 0));
        continue;
      }
      item.life -= dt;
      item.v.y -= this.gravity * dt;
      item.p.addScaledVector(item.v, dt);
      item.rot.addScaledVector(item.spin, dt);
      this.dummy.position.copy(item.p);
      this.dummy.rotation.set(item.rot.x, item.rot.y, item.rot.z);
      const age = 1 - item.life / item.max;
      const s = item.size * (1 + grow * age) * Math.min(1, item.life * 4);
      this.dummy.scale.setScalar(Math.max(s, 0.0001));
      this.dummy.updateMatrix();
      this.mesh.setMatrixAt(i, this.dummy.matrix);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  private hideAll(): void {
    const zero = new Matrix4().makeScale(0, 0, 0);
    for (let i = 0; i < this.count; i++) this.mesh.setMatrixAt(i, zero);
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}

export function coins(count: number): ShadowSpray {
  return new ShadowSpray(count, null, 1, 0.4, 3.2, goldCoin);
}

export function lampCage(act: Act, density: number, speed: number): void {
  const texture = latticeTexture();
  texture.repeat.set(3, 1);
  const shell = new Mesh(new CylinderGeometry(0.42, 0.42, 1.2, 10, 1, true), new MeshStandardMaterial({ map: texture, alphaTest: 0.5, side: DoubleSide, metalness: 0.75, roughness: 0.3 }));
  castShadow(shell, density, 0.18);
  act.free(shell);
  act.onUpdate((t) => {
    shell.position.copy(lampUniform.value);
    shell.rotation.y = t * speed;
  });
}
