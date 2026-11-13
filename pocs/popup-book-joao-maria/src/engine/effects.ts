import { AdditiveBlending, Color, NormalBlending, Sprite, SpriteMaterial, type Texture, Vector3 } from "three";

export interface Particle {
  sprite: Sprite;
  velocity: Vector3;
  age: number;
  life: number;
  size: number;
  spin: number;
}

export interface ParticleOptions {
  texture: Texture;
  count: number;
  rate: number;
  spawn: (p: Particle, rand: () => number) => void;
  gravity?: number;
  grow?: number;
  additive?: boolean;
  colors?: string[];
  drag?: number;
}

export class Particles {
  readonly sprites: Sprite[] = [];
  emitting = true;
  private readonly particles: Particle[] = [];
  private budget = 0;

  constructor(private readonly options: ParticleOptions) {
    for (let i = 0; i < options.count; i++) {
      const material = new SpriteMaterial({
        map: options.texture,
        transparent: true,
        depthWrite: false,
        blending: options.additive ? AdditiveBlending : NormalBlending,
      });
      const sprite = new Sprite(material);
      sprite.visible = false;
      this.sprites.push(sprite);
      this.particles.push({ sprite, velocity: new Vector3(), age: 0, life: 0, size: 0, spin: 0 });
    }
  }

  burst(amount: number): void {
    for (let i = 0; i < amount; i++) this.spawnOne();
  }

  update(dt: number): void {
    if (this.emitting) {
      this.budget += this.options.rate * dt;
      while (this.budget >= 1) {
        this.budget -= 1;
        this.spawnOne();
      }
    }
    const gravity = this.options.gravity ?? 0;
    const drag = this.options.drag ?? 0;
    for (const p of this.particles) {
      if (!p.sprite.visible) continue;
      p.age += dt;
      if (p.age >= p.life) {
        p.sprite.visible = false;
        continue;
      }
      p.velocity.y -= gravity * dt;
      p.velocity.multiplyScalar(1 - drag * dt);
      p.sprite.position.addScaledVector(p.velocity, dt);
      const t = p.age / p.life;
      const scale = p.size * (1 + (this.options.grow ?? 0) * t);
      p.sprite.scale.set(scale, scale, scale);
      p.sprite.material.rotation += p.spin * dt;
      p.sprite.material.opacity = Math.min(t * 6, 1) * (1 - Math.pow(t, 2));
    }
  }

  private spawnOne(): void {
    const p = this.particles.find((candidate) => !candidate.sprite.visible);
    if (!p) return;
    p.age = 0;
    p.spin = 0;
    p.velocity.set(0, 0, 0);
    this.options.spawn(p, Math.random);
    const colors = this.options.colors;
    if (colors) p.sprite.material.color = new Color(colors[Math.floor(Math.random() * colors.length)]);
    p.sprite.scale.setScalar(p.size);
    p.sprite.visible = true;
  }
}
