import { BufferAttribute, BufferGeometry, DoubleSide, Group, Mesh, MeshStandardMaterial, type Scene, Vector3 } from "three";
import { toTexture } from "../art/canvas";
import { paintLeaf, paintPuff } from "../art/props";
import { type Beanstalk, beanstalk } from "../scenes/models";
import { STALK_BASE, STALK_GIRTH } from "../scenes/stations";
import { STALK_HEIGHT } from "../scenes/timelines";
import { Particles } from "./effects";

const SPOTS = Array.from({ length: 30 }, (_, i) => 0.04 + i * 0.031);
const FALL_ANGLE = 1.48;

interface Bird {
  body: Group;
  left: Group;
  right: Group;
  radius: number;
  height: number;
  speed: number;
  phase: number;
}

function wing(side: number, material: MeshStandardMaterial): Group {
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(new Float32Array([0, 0, -0.12, 0, 0, 0.12, side * 0.55, 0, 0]), 3));
  geometry.computeVertexNormals();
  const pivot = new Group();
  pivot.add(new Mesh(geometry, material));
  return pivot;
}

function flock(): Bird[] {
  const material = new MeshStandardMaterial({ color: "#2b2b3a", side: DoubleSide, roughness: 0.8 });
  return Array.from({ length: 14 }, (_, i) => {
    const body = new Group();
    const left = wing(-1, material);
    const right = wing(1, material);
    const shape = new BufferGeometry().setAttribute("position", new BufferAttribute(new Float32Array([0, 0, -0.25, -0.06, 0, 0.1, 0.06, 0, 0.1]), 3));
    shape.computeVertexNormals();
    const torso = new Mesh(shape, material);
    body.add(left, right, torso);
    body.scale.setScalar(1.4);
    return { body, left, right, radius: 7 + (i % 4) * 1.4, height: 9 + i * 1.2, speed: 0.35 + (i % 3) * 0.08, phase: i * 0.9 };
  });
}

export class StalkWorld {
  private readonly stalk: Beanstalk;
  private readonly birds = flock();
  private readonly dirt: Particles;
  private readonly leaves: Particles;
  private readonly dust: Particles;
  private tip = new Vector3();
  private readonly pads: { group: Group; at: number }[] = [];
  private burst = false;

  constructor(scene: Scene) {
    this.stalk = beanstalk(STALK_HEIGHT, SPOTS, STALK_GIRTH);
    this.stalk.group.position.set(STALK_BASE.x, STALK_BASE.y, STALK_BASE.z);
    this.stalk.group.traverse((child) => {
      child.castShadow = true;
      child.receiveShadow = true;
    });
    scene.add(this.stalk.group);
    for (const bird of this.birds) scene.add(bird.body);
    const puff = toTexture(paintPuff());
    this.dirt = new Particles({
      texture: puff,
      count: 70,
      rate: 0,
      gravity: 4,
      grow: 1.6,
      colors: ["#8a5a2a", "#6e4220", "#a8743a"],
      spawn: (p, rand) => {
        const a = rand() * Math.PI * 2;
        p.sprite.position.set(STALK_BASE.x + Math.cos(a) * 0.8, 0.3, STALK_BASE.z + Math.sin(a) * 0.8);
        p.velocity.set(Math.cos(a) * (2 + rand() * 2), 3 + rand() * 4, Math.sin(a) * (2 + rand() * 2));
        p.life = 1.2 + rand() * 0.6;
        p.size = 0.6 + rand() * 0.5;
      },
    });
    this.leaves = new Particles({
      texture: toTexture(paintLeaf()),
      count: 80,
      rate: 0,
      gravity: 0.6,
      drag: 0.4,
      spawn: (p, rand) => {
        const a = rand() * Math.PI * 2;
        p.sprite.position.set(this.tip.x + Math.cos(a) * 1.4, this.tip.y, this.tip.z + Math.sin(a) * 1.4);
        p.velocity.set(-Math.sin(a) * 3, 1 + rand(), Math.cos(a) * 3);
        p.life = 2 + rand();
        p.size = 0.45 + rand() * 0.3;
        p.spin = (rand() - 0.5) * 6;
      },
    });
    this.dust = new Particles({
      texture: puff,
      count: 180,
      rate: 0,
      grow: 2.5,
      drag: 1.2,
      colors: ["#e8dcc0", "#d8ccb0", "#c8bca0"],
      spawn: (p, rand) => {
        const along = rand() * STALK_HEIGHT;
        p.sprite.position.set(STALK_BASE.x + (rand() - 0.5) * 3, 0.5, STALK_BASE.z - along);
        p.velocity.set((rand() - 0.5) * 8, 2 + rand() * 4, (rand() - 0.5) * 4);
        p.life = 2.5 + rand() * 1.5;
        p.size = 1.6 + rand() * 1.4;
      },
    });
    for (const system of [this.dirt, this.leaves, this.dust]) for (const sprite of system.sprites) scene.add(sprite);
  }

  attach(pad: Group, at: Vector3): void {
    pad.position.copy(at).sub(new Vector3(STALK_BASE.x, STALK_BASE.y, STALK_BASE.z));
    this.stalk.group.add(pad);
    this.pads.push({ group: pad, at: at.y / STALK_HEIGHT });
  }

  crash(): void {
    this.dust.burst(160);
  }

  update(visible: boolean, amount: number, fall: number, dt: number): void {
    this.stalk.group.visible = visible;
    this.stalk.grow(amount, (spot) => Math.min(Math.max((amount - spot) / 0.05, 0), 1));
    this.stalk.group.rotation.x = -fall * FALL_ANGLE;
    this.tip.copy(this.stalk.tip()).add(this.stalk.group.position);
    for (const pad of this.pads) {
      const grown = Math.min(Math.max((amount - pad.at) / 0.06, 0), 1);
      pad.group.visible = grown > 0;
      pad.group.scale.setScalar(Math.max(grown, 0.001));
    }
    if (visible && amount > 0 && amount < 0.2 && !this.burst) {
      this.dirt.burst(60);
      this.burst = true;
    }
    if (amount === 0) this.burst = false;
    this.leaves.options.rate = visible && amount > 0.02 && amount < 0.999 ? 26 : 0;
    for (const system of [this.dirt, this.leaves, this.dust]) system.update(dt);
    const flying = visible && fall === 0 && amount > 0.3;
    this.birds.forEach((bird) => {
      bird.body.visible = flying && bird.height < amount * STALK_HEIGHT;
      const a = performance.now() * 0.001 * bird.speed + bird.phase;
      bird.body.position.set(STALK_BASE.x + Math.cos(a) * bird.radius, bird.height + Math.sin(a * 2) * 0.6, STALK_BASE.z + Math.sin(a) * bird.radius);
      bird.body.rotation.y = -a;
      const flap = Math.sin(performance.now() * 0.018 + bird.phase) * 0.7;
      bird.left.rotation.z = flap;
      bird.right.rotation.z = -flap;
    });
  }
}
