import {
  AdditiveBlending,
  type CanvasTexture,
  BufferAttribute,
  BufferGeometry,
  Color,
  DirectionalLight,
  FogExp2,
  HemisphereLight,
  InstancedMesh,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PlaneGeometry,
  Points,
  PointsMaterial,
  RepeatWrapping,
  type Scene,
  SphereGeometry,
  SRGBColorSpace,
  Texture,
} from "three";
import { paint, seeded, toTexture } from "../art/canvas";
import { paintTree } from "../art/props";
import { CLOUD_Y, SKY_Y, STALK_BASE } from "../scenes/stations";
import type { Vec } from "./flight";
import { cutoutMesh } from "./popup";

const GROUND_SKY = new Color("#9fd6ff");
const HIGH_SKY = new Color("#2f5fd0");
const NIGHT_SKY = new Color("#0e1233");

export function skyColor(altitude: number, night: number): Color {
  const high = Math.min(Math.max(altitude / SKY_Y, 0), 1);
  return GROUND_SKY.clone().lerp(HIGH_SKY, high).lerp(NIGHT_SKY, night);
}

function streakTexture(): CanvasTexture {
  return toTexture(
    paint(256, 16, (ctx, w, h) => {
      const g = ctx.createLinearGradient(0, 0, w, 0);
      g.addColorStop(0, "rgba(255,255,255,0)");
      g.addColorStop(1, "rgba(255,250,220,1)");
      ctx.fillStyle = g;
      ctx.fillRect(0, h / 2 - 3, w, 6);
    }),
  );
}

function groundTexture(grass: HTMLImageElement): Texture {
  const texture = new Texture(grass);
  texture.colorSpace = SRGBColorSpace;
  texture.wrapS = RepeatWrapping;
  texture.wrapT = RepeatWrapping;
  texture.repeat.set(60, 60);
  texture.anisotropy = 8;
  texture.needsUpdate = true;
  return texture;
}

function cloudSea(): InstancedMesh {
  const rand = seeded(21);
  const clusters = 110;
  const puffs = 6;
  const mesh = new InstancedMesh(
    new SphereGeometry(1, 16, 12),
    new MeshStandardMaterial({ color: "#ffffff", roughness: 1 }),
    clusters * puffs,
  );
  const m = new Matrix4();
  let i = 0;
  for (let c = 0; c < clusters; c++) {
    let x = -40 + rand() * 160;
    let z = -70 + rand() * 80;
    if (Math.hypot(x - STALK_BASE.x, z - STALK_BASE.z) < 7) x += 14;
    if (z > -14 && x > 26 && x < 80) z -= 28;
    const size = 2 + rand() * 2.5;
    for (let p = 0; p < puffs; p++) {
      const r = size * (0.6 + rand() * 0.5);
      m.makeScale(r * 1.3, r * 0.7, r);
      m.setPosition(x + (p - 2.5) * size * 0.7, CLOUD_Y + rand() * 1.2, z + (rand() - 0.5) * size);
      mesh.setMatrixAt(i, m);
      i += 1;
    }
  }
  mesh.receiveShadow = true;
  return mesh;
}

function starDome(): Points<BufferGeometry, PointsMaterial> {
  const rand = seeded(5);
  const count = 900;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const a = rand() * Math.PI * 2;
    const h = 0.15 + rand() * 0.85;
    const r = 180;
    positions[i * 3] = 40 + Math.cos(a) * r * Math.sqrt(1 - h * h);
    positions[i * 3 + 1] = h * r;
    positions[i * 3 + 2] = Math.sin(a) * r * Math.sqrt(1 - h * h);
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(positions, 3));
  return new Points(geometry, new PointsMaterial({ color: "#fff6d8", size: 1.1, transparent: true, opacity: 0, depthWrite: false, blending: AdditiveBlending, fog: false }));
}

export class World {
  private readonly sun = new DirectionalLight("#fff1d6", 2.6);
  private readonly hemi = new HemisphereLight("#ffe9c7", "#3a2414", 1.1);
  private readonly stars = starDome();
  private readonly fog = new FogExp2("#9fd6ff", 0.012);
  private readonly meteors: Mesh<PlaneGeometry, MeshBasicMaterial>[] = [];

  constructor(private readonly scene: Scene, grass: HTMLImageElement) {
    scene.fog = this.fog;
    scene.background = GROUND_SKY.clone();
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    Object.assign(this.sun.shadow.camera, { left: -9, right: 9, top: 9, bottom: -9, far: 60 });
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.02;
    scene.add(this.sun, this.sun.target, this.hemi, this.stars, cloudSea());
    const ground = new Mesh(new PlaneGeometry(500, 500), new MeshStandardMaterial({ map: groundTexture(grass), color: "#c8e0a0", roughness: 0.95 }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(40, -0.01, 0);
    ground.receiveShadow = true;
    scene.add(ground);
    this.forest();
    for (let i = 0; i < 3; i++) {
      const meteor = new Mesh(new PlaneGeometry(9, 0.18), new MeshBasicMaterial({ map: streakTexture(), transparent: true, blending: AdditiveBlending, depthWrite: false, fog: false }));
      this.meteors.push(meteor);
      scene.add(meteor);
    }
  }

  update(look: Vec, altitude: number, night: number, time: number): void {
    const starry = Math.max(night, Math.min(Math.max((altitude - 20) / SKY_Y, 0), 0.6));
    this.meteors.forEach((meteor, i) => {
      const cycle = (time * 0.33 + i * 0.37) % 1;
      meteor.visible = starry > 0.3 && cycle < 0.25;
      const p = cycle / 0.25;
      meteor.position.set(look.x - 40 + i * 25 + p * 50, look.y + 30 - p * 14 + i * 6, look.z - 60);
      meteor.rotation.z = -0.27;
      meteor.material.opacity = Math.sin(p * Math.PI);
    });
    const color = skyColor(altitude, night);
    (this.scene.background as Color).copy(color);
    this.fog.color.copy(color);
    this.fog.density = 0.012 * (1 - Math.min(altitude / SKY_Y, 1) * 0.5);
    this.sun.target.position.set(look.x, look.y, look.z);
    this.sun.position.set(look.x - 5, look.y + 11, look.z + 7);
    this.sun.intensity = 2.6 - night * 2.1;
    this.sun.color.set(night > 0.5 ? "#9fb0ff" : "#fff1d6");
    this.hemi.intensity = 1.1 - night * 0.6;
    this.stars.material.opacity = starry;
  }

  private forest(): void {
    const rand = seeded(77);
    for (let i = 0; i < 46; i++) {
      const tree = cutoutMesh(paintTree(i * 13 + 5, rand() > 0.7), 3 + rand() * 3);
      const x = -20 + rand() * 110;
      const z = -10 - rand() * 30;
      if (Math.abs(x - STALK_BASE.x) < 5 && z > -14) continue;
      tree.position.set(x, 0, z);
      tree.rotation.y = (rand() - 0.5) * 0.4;
      this.scene.add(tree);
    }
  }
}
