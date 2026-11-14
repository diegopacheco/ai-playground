import {
  AdditiveBlending,
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  Color,
  CylinderGeometry,
  DoubleSide,
  ExtrudeGeometry,
  Group,
  HemisphereLight,
  LatheGeometry,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Path,
  PlaneGeometry,
  PointLight,
  Points,
  PointsMaterial,
  RectAreaLight,
  Shape,
  Sprite,
  SpriteMaterial,
  type Texture,
  TorusGeometry,
  Vector2,
  Vector3,
} from "three";
import { seeded } from "../art/canvas";
import { audienceTexture, paperTexture, rugTexture, woodTexture } from "../art/textures";
import { lampUniform, screenMaterial } from "./materials";
import { SCREEN_H, SCREEN_W } from "./shadow";

const FLOOR_Y = -1.4;
const MOTES = 260;

function starPath(x: number, y: number, r: number): Path {
  const path = new Path();
  for (let i = 0; i < 16; i++) {
    const a = (i * Math.PI) / 8 - Math.PI / 2;
    const rr = i % 2 === 0 ? r : r * 0.55;
    if (i === 0) path.moveTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr);
    else path.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr);
  }
  path.closePath();
  return path;
}

function frameShape(): Shape {
  const shape = new Shape();
  shape.moveTo(-4.9, -0.55);
  shape.lineTo(4.9, -0.55);
  shape.lineTo(4.9, 5.45);
  shape.bezierCurveTo(4.9, 6.2, 1.2, 6.0, 0.35, 6.95);
  shape.quadraticCurveTo(0, 7.25, 0, 7.6);
  shape.quadraticCurveTo(0, 7.25, -0.35, 6.95);
  shape.bezierCurveTo(-1.2, 6.0, -4.9, 6.2, -4.9, 5.45);
  shape.closePath();
  const hole = new Path();
  hole.moveTo(-SCREEN_W / 2, 0);
  hole.lineTo(SCREEN_W / 2, 0);
  hole.lineTo(SCREEN_W / 2, SCREEN_H);
  hole.lineTo(-SCREEN_W / 2, SCREEN_H);
  hole.closePath();
  shape.holes.push(hole, starPath(0, 5.95, 0.5), starPath(-2.6, 5.35, 0.28), starPath(2.6, 5.35, 0.28));
  return shape;
}

function brassLamp(brass: MeshStandardMaterial): Group {
  const lamp = new Group();
  const points = [
    [0, -0.5],
    [0.16, -0.5],
    [0.1, -0.42],
    [0.06, -0.3],
    [0.24, -0.16],
    [0.28, -0.08],
    [0.2, 0],
    [0.04, 0.02],
  ].map(([x, y]) => new Vector2(x, y));
  const body = new Mesh(new LatheGeometry(points, 28), brass);
  body.position.y = -0.1;
  const spout = new Mesh(new CylinderGeometry(0.035, 0.08, 0.42, 12), brass);
  spout.rotation.z = -1.15;
  spout.position.set(0.3, -0.08, 0);
  const handle = new Mesh(new TorusGeometry(0.12, 0.022, 8, 20, Math.PI * 1.2), brass);
  handle.position.set(-0.3, -0.1, 0);
  handle.rotation.z = 1.2;
  lamp.add(body, spout, handle);
  lamp.position.set(-0.46, -0.04, 0);
  return lamp;
}

export class Room {
  readonly group = new Group();
  readonly screen: Mesh;
  private readonly top = new Color("#ffb45c");
  private readonly bottom = new Color("#c2410c");
  private readonly area: RectAreaLight;
  private readonly lampLight: PointLight;
  private readonly lanterns: PointLight[] = [];
  private readonly flame: Sprite;
  private readonly core: Sprite;
  private readonly rig = new Group();
  private readonly glass: Mesh<PlaneGeometry, MeshStandardMaterial>;
  private readonly glow: MeshBasicMaterial;
  private readonly motes: Points<BufferGeometry, PointsMaterial>;
  private readonly moteSeeds: Float32Array;

  constructor(mask: Texture, spark: Texture) {
    const wood = new MeshStandardMaterial({ map: woodTexture(), color: "#d79a62", roughness: 0.5, metalness: 0.05 });
    const gold = new MeshStandardMaterial({ color: "#d9a441", roughness: 0.28, metalness: 0.92 });
    const brass = new MeshStandardMaterial({ color: "#c8902f", roughness: 0.32, metalness: 0.9 });

    this.screen = new Mesh(new PlaneGeometry(SCREEN_W, SCREEN_H), screenMaterial(mask, paperTexture(), this.top, this.bottom));
    this.screen.position.set(0, SCREEN_H / 2, 0);
    this.group.add(this.screen);

    const frame = new Mesh(new ExtrudeGeometry(frameShape(), { depth: 0.36, bevelEnabled: true, bevelSize: 0.06, bevelThickness: 0.06, bevelSegments: 2 }), wood);
    frame.position.z = 0.02;
    this.group.add(frame);

    this.glow = new MeshBasicMaterial({ color: "#ffb45c" });
    for (const [x, y, r] of [
      [0, 5.95, 0.55],
      [-2.6, 5.35, 0.32],
      [2.6, 5.35, 0.32],
    ] as const) {
      const glowPanel = new Mesh(new PlaneGeometry(r * 2.2, r * 2.2), this.glow);
      glowPanel.position.set(x, y, -0.02);
      this.group.add(glowPanel);
    }

    const trims: [number, number, number, number][] = [
      [0, -0.06, SCREEN_W + 0.24, 0.12],
      [0, SCREEN_H + 0.06, SCREEN_W + 0.24, 0.12],
      [-SCREEN_W / 2 - 0.06, SCREEN_H / 2, 0.12, SCREEN_H],
      [SCREEN_W / 2 + 0.06, SCREEN_H / 2, 0.12, SCREEN_H],
    ];
    for (const [x, y, w, h] of trims) {
      const trim = new Mesh(new BoxGeometry(w, h, 0.1), gold);
      trim.position.set(x, y, 0.45);
      this.group.add(trim);
    }

    const dome = new LatheGeometry(
      [
        [0, 0],
        [0.42, 0],
        [0.5, 0.25],
        [0.36, 0.6],
        [0.12, 0.85],
        [0.03, 1.05],
        [0, 1.15],
      ].map(([x, y]) => new Vector2(x, y)),
      24,
    );
    for (const side of [-1, 1]) {
      const pillar = new Mesh(new CylinderGeometry(0.3, 0.36, 7.2, 20), wood);
      pillar.position.set(side * 5.35, 2.2, 0.3);
      this.group.add(pillar);
      for (const y of [-1.0, 0.3, 4.4, 5.75]) {
        const ring = new Mesh(new TorusGeometry(0.34, 0.06, 10, 28), gold);
        ring.rotation.x = Math.PI / 2;
        ring.position.set(side * 5.35, y, 0.3);
        this.group.add(ring);
      }
      const cap = new Mesh(dome, gold);
      cap.position.set(side * 5.35, 5.8, 0.3);
      this.group.add(cap);
      this.group.add(this.lantern(side * 6.6, 5.4, 1.8, brass));
    }

    const base = new Mesh(new BoxGeometry(11.4, 0.85, 1.2), wood);
    base.position.set(0, FLOOR_Y + 0.42, 0.2);
    this.group.add(base);

    const floor = new Mesh(new PlaneGeometry(60, 40), new MeshStandardMaterial({ color: "#1d0f0a", roughness: 0.9 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = FLOOR_Y;
    this.group.add(floor);
    const rug = new Mesh(new PlaneGeometry(11, 6.2), new MeshStandardMaterial({ map: rugTexture(), roughness: 0.95 }));
    rug.rotation.x = -Math.PI / 2;
    rug.position.set(0, FLOOR_Y + 0.01, 4.4);
    this.group.add(rug);

    const audience = new Mesh(new PlaneGeometry(22, 2.75), new MeshBasicMaterial({ map: audienceTexture(), color: "#000000", alphaTest: 0.5, side: DoubleSide }));
    audience.position.set(0, -1.05, 8.6);
    this.group.add(audience);

    const backWall = new Mesh(new PlaneGeometry(40, 20), new MeshStandardMaterial({ color: "#140a10", roughness: 1 }));
    backWall.position.set(0, 6, -9);
    this.group.add(backWall);

    const stand = new Mesh(new CylinderGeometry(0.08, 0.14, 1, 12), wood);
    stand.position.y = -0.98;
    const plate = new Mesh(new CylinderGeometry(0.42, 0.42, 0.06, 24), brass);
    plate.position.y = -0.5;
    this.rig.add(stand, plate, brassLamp(brass));
    this.flame = new Sprite(new SpriteMaterial({ map: spark, color: "#ffb347", blending: AdditiveBlending, depthWrite: false, transparent: true }));
    this.core = new Sprite(new SpriteMaterial({ map: spark, color: "#fff4d6", blending: AdditiveBlending, depthWrite: false, transparent: true }));
    this.rig.add(this.flame, this.core);
    this.glass = new Mesh(new PlaneGeometry(1.1, 1.3), new MeshStandardMaterial({ color: "#ffb45c", emissive: "#ffb45c", emissiveIntensity: 0.6, transparent: true, opacity: 0.55, side: DoubleSide, roughness: 0.1 }));
    this.glass.position.set(0, 0.05, 0.7);
    const glassFrame = new Mesh(new TorusGeometry(0.72, 0.035, 8, 4), brass);
    glassFrame.rotation.z = Math.PI / 4;
    glassFrame.scale.set(0.8, 0.93, 1);
    glassFrame.position.copy(this.glass.position);
    this.rig.add(this.glass, glassFrame);
    this.lampLight = new PointLight("#ffb45c", 8, 14, 1.6);
    this.rig.add(this.lampLight);
    this.group.add(this.rig);

    this.area = new RectAreaLight("#ffb45c", 3, SCREEN_W, SCREEN_H);
    this.area.position.set(0, SCREEN_H / 2, 0.1);
    this.area.lookAt(0, SCREEN_H / 2, 10);
    this.group.add(this.area);
    this.group.add(new HemisphereLight("#5a4275", "#1c0e08", 0.6));
    const fill = new PointLight("#ffb070", 18, 22, 1.5);
    fill.position.set(0, 4.5, 7);
    this.group.add(fill);

    const rand = seeded(9);
    const positions = new Float32Array(MOTES * 3);
    this.moteSeeds = new Float32Array(MOTES * 3);
    for (let i = 0; i < MOTES; i++) {
      this.moteSeeds[i * 3] = (rand() - 0.5) * 9;
      this.moteSeeds[i * 3 + 1] = rand() * 6;
      this.moteSeeds[i * 3 + 2] = 0.3 + rand() * 5;
    }
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(positions, 3));
    this.motes = new Points(geometry, new PointsMaterial({ map: spark, size: 0.06, transparent: true, blending: AdditiveBlending, depthWrite: false, color: "#ffcf8a" }));
    this.group.add(this.motes);
  }

  setTint(top: Color, bottom: Color): void {
    this.top.copy(top);
    this.bottom.copy(bottom);
  }

  update(time: number, level: number): void {
    const flicker = 1 + Math.sin(time * 17) * 0.03 + Math.sin(time * 7.3) * 0.04 + Math.sin(time * 31) * 0.02;
    const lit = level * flicker;
    const uniforms = (this.screen.material as unknown as { uniforms: { level: { value: number } } }).uniforms;
    uniforms.level.value = lit;
    const mid = this.top.clone().lerp(this.bottom, 0.5);
    this.area.color.copy(mid);
    this.area.intensity = lit * 4.2;
    this.glow.color.copy(this.top).multiplyScalar(lit * 1.4);
    this.glass.material.color.copy(mid);
    this.glass.material.emissive.copy(mid);
    this.glass.material.emissiveIntensity = 0.3 + lit * 0.8;
    this.lampLight.color.copy(this.top).lerp(new Color("#ffb347"), 0.5);
    this.lampLight.intensity = lit * 9;
    const lamp = lampUniform.value;
    this.rig.position.copy(lamp);
    const size = Math.max(level, 0.02);
    this.flame.scale.set(0.5 * size * flicker, 0.85 * size * flicker, 1);
    this.flame.position.y = 0.1 * size;
    this.core.scale.set(0.18 * size, 0.32 * size, 1);
    this.core.position.y = 0.06 * size;
    for (const [i, light] of this.lanterns.entries()) light.intensity = 2.2 + Math.sin(time * 9 + i * 2) * 0.25 + Math.sin(time * 23 + i) * 0.15;
    const positions = this.motes.geometry.attributes.position as BufferAttribute;
    for (let i = 0; i < MOTES; i++) {
      const sx = this.moteSeeds[i * 3] ?? 0;
      const sy = this.moteSeeds[i * 3 + 1] ?? 0;
      const sz = this.moteSeeds[i * 3 + 2] ?? 0;
      positions.setXYZ(i, sx + Math.sin(time * 0.21 + i) * 0.35, ((sy + time * 0.07 * (1 + (i % 5) * 0.2)) % 6.2) - 0.4, sz + Math.cos(time * 0.17 + i * 1.3) * 0.3);
    }
    positions.needsUpdate = true;
    this.motes.material.color.copy(mid).multiplyScalar(0.5 + lit * 0.9);
  }

  private lantern(x: number, y: number, z: number, brass: MeshStandardMaterial): Group {
    const group = new Group();
    const chain = new Mesh(new CylinderGeometry(0.015, 0.015, 3, 6), brass);
    chain.position.y = 1.9;
    const body = new Mesh(
      new LatheGeometry(
        [
          [0, -0.45],
          [0.08, -0.42],
          [0.26, -0.15],
          [0.3, 0.1],
          [0.18, 0.32],
          [0.05, 0.4],
          [0, 0.42],
        ].map(([a, b]) => new Vector2(a, b)),
        8,
      ),
      new MeshStandardMaterial({ color: "#ffae42", emissive: "#ff8a1f", emissiveIntensity: 2.4, roughness: 0.4 }),
    );
    const capTop = new Mesh(new CylinderGeometry(0.06, 0.24, 0.16, 8), brass);
    capTop.position.y = 0.4;
    const light = new PointLight("#ff9a3c", 2.2, 6, 1.8);
    this.lanterns.push(light);
    group.add(chain, body, capTop, light);
    group.position.set(x, y, z);
    return group;
  }
}

export function lampTarget(rest: Vector3, motion: Vector3, pointer: Vector3): Vector3 {
  return new Vector3(rest.x + motion.x + pointer.x, rest.y + motion.y + pointer.y, Math.min(rest.z + motion.z + pointer.z, -1.6));
}
