import {
  AdditiveBlending,
  BoxGeometry,
  Color,
  ConeGeometry,
  CylinderGeometry,
  DoubleSide,
  Group,
  HemisphereLight,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PlaneGeometry,
  RepeatWrapping,
  SphereGeometry,
  SpotLight,
  Vector3,
} from "three";
import { toTexture } from "../art/canvas";
import { paintBeam, paintDamask, paintFrame, paintHead, paintMarquee, paintPlanks, paintSkirt, paintValance, paintVelvet } from "../art/theater";
import { FLOOR_Y, type Mood, STAGE_D, STAGE_W } from "./popup";

export const OPENING_H = 5.2;
export const BACKDROP_Z = -STAGE_D / 2 - 0.05;
export const BACKDROP_W = STAGE_W + 0.6;
export const BACKDROP_H = OPENING_H + 0.3;
const CURTAIN_Z = STAGE_D / 2 - 0.15;
const CURTAIN_TOP = FLOOR_Y + OPENING_H + 0.3;
const CURTAIN_H = CURTAIN_TOP - FLOOR_Y + 0.02;
const HALF_W = STAGE_W / 2 + 0.35;
const SEG_X = 56;
const SEG_Y = 18;
const PLEATS = 11;

function openWidth(v: number): number {
  return v > 0.35 ? 0.55 + ((v - 0.35) / 0.65) * 0.65 : 0.55 + ((0.35 - v) / 0.35) * 0.3;
}

class CurtainHalf {
  readonly mesh: Mesh<PlaneGeometry, MeshStandardMaterial>;

  constructor(private readonly side: -1 | 1, material: MeshStandardMaterial) {
    this.mesh = new Mesh(new PlaneGeometry(1, 1, SEG_X, SEG_Y), material);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
  }

  shape(closed: number, time: number, speed: number): void {
    const position = this.mesh.geometry.attributes.position!;
    const uv = this.mesh.geometry.attributes.uv!;
    for (let i = 0; i < position.count; i++) {
      const u = this.side < 0 ? uv.getX(i) : 1 - uv.getX(i);
      const v = uv.getY(i);
      const width = HALF_W + (openWidth(v) - HALF_W) * (1 - closed);
      const sway = Math.sin(time * 1.1 + v * 3 + this.side) * 0.012 * (1 - v);
      const x = this.side * (HALF_W + 0.05 - u * width) + sway;
      const amp = 0.06 + 0.1 * (1 - closed);
      const billow = Math.sin(Math.PI * u) * (1 - v) * speed * 0.35;
      const z = CURTAIN_Z + Math.sin(u * PLEATS * Math.PI * 2 + this.side) * amp * (0.6 + 0.4 * v) + billow;
      position.setXYZ(i, x, FLOOR_Y + v * CURTAIN_H - 0.01, z);
    }
    position.needsUpdate = true;
    this.mesh.geometry.computeVertexNormals();
    this.mesh.geometry.computeBoundingSphere();
  }
}

function box(w: number, h: number, d: number, material: MeshStandardMaterial | MeshStandardMaterial[], x: number, y: number, z: number): Mesh {
  const mesh = new Mesh(new BoxGeometry(w, h, d), material);
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function beam(light: SpotLight, length: number, radius: number): Mesh {
  const geometry = new ConeGeometry(radius, length, 40, 1, true);
  geometry.translate(0, -length / 2, 0);
  const material = new MeshBasicMaterial({ map: toTexture(paintBeam()), transparent: true, opacity: 0.05, depthWrite: false, blending: AdditiveBlending, side: DoubleSide, color: light.color });
  const mesh = new Mesh(geometry, material);
  mesh.position.copy(light.position);
  return mesh;
}

function aimBeam(cone: Mesh, from: Vector3, to: Vector3): void {
  const dir = to.clone().sub(from).normalize();
  cone.quaternion.setFromUnitVectors(new Vector3(0, -1, 0), dir);
}

export class Theater {
  readonly group = new Group();
  readonly backdrop: MeshStandardMaterial;
  private readonly halves: CurtainHalf[];
  private readonly spots: SpotLight[] = [];
  private readonly cones: Mesh[] = [];
  private readonly foot: SpotLight;
  private readonly hemi: HemisphereLight;
  private readonly bulbs: MeshStandardMaterial[] = [];
  private readonly heads: Mesh[] = [];
  private readonly aim = new Vector3(0, 2, 0);
  private closed = 0;

  constructor() {
    const velvet = toTexture(paintVelvet());
    const curtainMaterial = new MeshStandardMaterial({ map: velvet, side: DoubleSide, roughness: 0.75, metalness: 0.05 });
    this.halves = [new CurtainHalf(-1, curtainMaterial), new CurtainHalf(1, curtainMaterial)];
    for (const half of this.halves) this.group.add(half.mesh);
    this.backdrop = new MeshStandardMaterial({ roughness: 0.95 });
    const drop = new Mesh(new PlaneGeometry(BACKDROP_W, BACKDROP_H), this.backdrop);
    drop.position.set(0, FLOOR_Y + BACKDROP_H / 2 - 0.02, BACKDROP_Z);
    drop.receiveShadow = true;
    this.group.add(drop);
    this.room();
    this.stageBox();
    this.proscenium();
    this.footlights();
    this.audience();
    this.hemi = new HemisphereLight("#ffe2c0", "#2a0a10", 0.5);
    this.group.add(this.hemi);
    for (const side of [-1, 1]) {
      const spot = new SpotLight("#ffe7c0", 26, 30, 0.2, 0.6, 1.4);
      spot.position.set(side * 5.5, 9.5, 9);
      spot.castShadow = true;
      spot.shadow.mapSize.set(1024, 1024);
      spot.shadow.bias = -0.0005;
      spot.shadow.normalBias = 0.02;
      this.group.add(spot, spot.target);
      this.spots.push(spot);
      const cone = beam(spot, 11, 1.6);
      this.cones.push(cone);
      this.group.add(cone);
    }
    this.foot = new SpotLight("#ffc98a", 14, 14, 0.75, 0.8, 1.2);
    this.foot.position.set(0, FLOOR_Y + 0.15, STAGE_D / 2 + 1.6);
    this.foot.target.position.set(0, FLOOR_Y + 2.6, -STAGE_D / 2);
    this.foot.castShadow = true;
    this.foot.shadow.mapSize.set(1024, 1024);
    this.foot.shadow.bias = -0.0005;
    this.group.add(this.foot, this.foot.target);
    this.shapeCurtains(0, 0, 0);
  }

  setBackdrop(texture: MeshStandardMaterial["map"]): void {
    this.backdrop.map = texture;
    this.backdrop.needsUpdate = true;
  }

  setCurtain(closed: number, time: number, dt: number): void {
    const speed = dt > 0 ? (closed - this.closed) / dt : 0;
    this.closed = closed;
    this.shapeCurtains(closed, time, speed);
  }

  update(time: number, mood: Mood, focus: Vector3): void {
    const target = this.closed > 0.6 ? new Vector3(0, FLOOR_Y + 2.4, CURTAIN_Z) : focus.clone().add(new Vector3(0, FLOOR_Y, 0));
    this.aim.lerp(target, 0.05);
    this.spots.forEach((spot, i) => {
      const wobble = new Vector3(Math.sin(time * 0.7 + i * 2) * 0.25, Math.cos(time * 0.5 + i) * 0.15, 0);
      spot.target.position.copy(this.aim).add(wobble).add(new Vector3((i === 0 ? -1 : 1) * 0.5, 0, 0));
      spot.color.lerp(new Color(i === 0 ? mood.key : mood.fill), 0.04);
      const cone = this.cones[i];
      if (cone) {
        aimBeam(cone, spot.position, spot.target.position);
        (cone.material as MeshBasicMaterial).color.copy(spot.color);
      }
    });
    this.foot.color.lerp(new Color(mood.fill), 0.04);
    this.hemi.intensity += (mood.ambient - this.hemi.intensity) * 0.04;
    const step = Math.floor(time * 6);
    this.bulbs.forEach((bulb, i) => {
      bulb.emissiveIntensity = (i + step) % 3 === 0 ? 3.2 : 0.7;
    });
    this.heads.forEach((head, i) => {
      head.position.y = head.userData.y + Math.abs(Math.sin(time * (1.3 + (i % 3) * 0.4) + i)) * 0.05;
      head.rotation.z = Math.sin(time * 0.6 + i * 1.7) * 0.06;
    });
  }

  private shapeCurtains(closed: number, time: number, speed: number): void {
    for (const half of this.halves) half.shape(closed, time, speed);
  }

  private room(): void {
    const damask = toTexture(paintDamask());
    damask.wrapS = RepeatWrapping;
    damask.wrapT = RepeatWrapping;
    damask.repeat.set(10, 4);
    const wall = new Mesh(new PlaneGeometry(40, 16), new MeshStandardMaterial({ map: damask, roughness: 0.9 }));
    wall.position.set(0, 8, -3.2);
    wall.receiveShadow = true;
    this.group.add(wall);
    const floor = new Mesh(new PlaneGeometry(40, 30), new MeshStandardMaterial({ color: "#3d0b14", roughness: 1 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.z = 8;
    floor.receiveShadow = true;
    this.group.add(floor);
  }

  private stageBox(): void {
    const planks = toTexture(paintPlanks());
    planks.wrapS = RepeatWrapping;
    planks.wrapT = RepeatWrapping;
    planks.repeat.set(2, 1.4);
    const skirt = toTexture(paintSkirt());
    skirt.wrapS = RepeatWrapping;
    skirt.repeat.set(3, 1);
    const top = new MeshStandardMaterial({ map: planks, roughness: 0.7 });
    const side = new MeshStandardMaterial({ map: skirt, roughness: 0.8 });
    const depth = STAGE_D + 0.9;
    this.group.add(box(STAGE_W + 1.8, FLOOR_Y, depth, [side, side, top, side, side, side], 0, FLOOR_Y / 2, 0.3));
  }

  private proscenium(): void {
    const pillar = new MeshStandardMaterial({ map: toTexture(paintFrame(256, 1024, true)), roughness: 0.6, metalness: 0.15 });
    const header = new MeshStandardMaterial({ map: toTexture(paintFrame(2048, 256, false)), roughness: 0.6, metalness: 0.15 });
    const plain = new MeshStandardMaterial({ color: "#d9bd86", roughness: 0.6 });
    const z = STAGE_D / 2 + 0.25;
    const h = OPENING_H + 1.6;
    for (const s of [-1, 1]) this.group.add(box(1.0, h, 0.6, [plain, plain, plain, plain, pillar, plain], s * (STAGE_W / 2 + 0.55), FLOOR_Y + h / 2 - 0.05, z));
    this.group.add(box(STAGE_W + 2.1, 1.1, 0.6, [plain, plain, plain, plain, header, plain], 0, FLOOR_Y + OPENING_H + 0.55, z));
    const valance = new Mesh(new PlaneGeometry(STAGE_W + 0.4, 1.0), new MeshStandardMaterial({ map: toTexture(paintValance()), alphaTest: 0.5, side: DoubleSide, roughness: 0.75 }));
    valance.position.set(0, FLOOR_Y + OPENING_H - 0.45, z - 0.32);
    valance.castShadow = true;
    this.group.add(valance);
    const sign = toTexture(paintMarquee("Pinocchio"));
    const marquee = new Mesh(new PlaneGeometry(4.6, 4.6 * (360 / 1400)), new MeshStandardMaterial({ map: sign, emissiveMap: sign, emissive: "#ffffff", emissiveIntensity: 0.55, alphaTest: 0.4, roughness: 0.5 }));
    marquee.position.set(0, FLOOR_Y + OPENING_H + 1.65, z + 0.05);
    this.group.add(marquee);
    const bulbGeometry = new SphereGeometry(0.06, 12, 8);
    const count = 26;
    for (let i = 0; i < count; i++) {
      const t = i / (count - 1);
      const angle = Math.PI * (1 - t);
      const material = new MeshStandardMaterial({ color: "#fff3c4", emissive: "#ffcf5a", emissiveIntensity: 1 });
      const bulb = new Mesh(bulbGeometry, material);
      bulb.position.set(Math.cos(angle) * 2.25, FLOOR_Y + OPENING_H + 1.45 + Math.sin(angle) * 0.62, z + 0.12);
      this.bulbs.push(material);
      this.group.add(bulb);
    }
    for (const s of [-1, 1]) {
      for (let i = 0; i < 9; i++) {
        const material = new MeshStandardMaterial({ color: "#fff3c4", emissive: "#ffcf5a", emissiveIntensity: 1 });
        const bulb = new Mesh(bulbGeometry, material);
        bulb.position.set(s * (STAGE_W / 2 + 0.55), FLOOR_Y + 0.5 + i * 0.68, z + 0.33);
        this.bulbs.push(material);
        this.group.add(bulb);
      }
    }
  }

  private footlights(): void {
    const shell = new MeshStandardMaterial({ color: "#c9932f", metalness: 0.7, roughness: 0.35, side: DoubleSide });
    const glow = new MeshStandardMaterial({ color: "#fff6d8", emissive: "#ffd27a", emissiveIntensity: 2.6 });
    const count = 11;
    for (let i = 0; i < count; i++) {
      const x = -STAGE_W / 2 + 0.3 + (i * (STAGE_W - 0.6)) / (count - 1);
      const hood = new Mesh(new CylinderGeometry(0.16, 0.16, 0.3, 16, 1, true, 0, Math.PI), shell);
      hood.rotation.z = Math.PI / 2;
      hood.rotation.y = Math.PI / 2;
      hood.position.set(x, FLOOR_Y + 0.08, STAGE_D / 2 + 0.6);
      const bulb = new Mesh(new SphereGeometry(0.07, 12, 8), glow);
      bulb.position.set(x, FLOOR_Y + 0.07, STAGE_D / 2 + 0.55);
      this.group.add(hood, bulb);
    }
  }

  private audience(): void {
    const material = (seed: number) => new MeshBasicMaterial({ map: toTexture(paintHead(seed)), alphaTest: 0.5, color: "#ffffff" });
    for (let row = 0; row < 2; row++) {
      for (let i = 0; i < 11; i++) {
        const head = new Mesh(new PlaneGeometry(0.95, 1.05), material(row * 31 + i * 7 + 1));
        const x = -6.2 + i * 1.24 + (row ? 0.6 : 0);
        head.userData.y = 0.15 + row * 0.25;
        head.position.set(x, head.userData.y + 0.52, 5.8 + (1 - row) * 1.2);
        this.heads.push(head);
        this.group.add(head);
      }
    }
  }
}
