import { type CanvasTexture, DoubleSide, Group, Mesh, MeshStandardMaterial, type Object3D, PlaneGeometry, type ShaderMaterial } from "three";
import { toTexture } from "../art/canvas";
import { MASK_LAYER, maskMaterial } from "./materials";

export interface SilOptions {
  anchorX?: number;
  anchorY?: number;
  density?: number;
  soft?: number;
}

export class Sil {
  readonly root = new Group();
  readonly width: number;
  readonly height: number;
  private readonly frames: CanvasTexture[];
  private readonly leather: MeshStandardMaterial;
  private readonly shade: ShaderMaterial;
  private frame = 0;

  constructor(canvases: HTMLCanvasElement | HTMLCanvasElement[], height: number, options: SilOptions = {}) {
    const list = Array.isArray(canvases) ? canvases : [canvases];
    const first = list[0];
    if (!first) throw new Error("a silhouette needs a canvas");
    this.frames = list.map(toTexture);
    this.height = height;
    this.width = (first.width / first.height) * height;
    const ax = options.anchorX ?? 0.5;
    const ay = options.anchorY ?? 0;
    const geometry = new PlaneGeometry(this.width, height);
    geometry.translate(this.width * (0.5 - ax), height * (0.5 - ay), 0);
    this.leather = new MeshStandardMaterial({ map: this.frames[0], alphaTest: 0.5, side: DoubleSide, roughness: 0.55, metalness: 0.1 });
    this.shade = maskMaterial(this.frames[0] ?? null, options.density ?? 1, options.soft ?? 1);
    const visible = new Mesh(geometry, this.leather);
    const shadow = new Mesh(geometry, this.shade);
    shadow.layers.set(MASK_LAYER);
    this.root.add(visible, shadow);
  }

  show(index: number): void {
    const next = ((index % this.frames.length) + this.frames.length) % this.frames.length;
    if (next === this.frame) return;
    this.frame = next;
    this.leather.map = this.frames[next] ?? null;
    this.shade.uniforms.map!.value = this.frames[next];
  }

  attach(child: Sil | Object3D, x: number, y: number, z = 0.004): void {
    const node = child instanceof Sil ? child.root : child;
    node.position.set(x, y, z);
    this.root.add(node);
  }

  set density(value: number) {
    this.shade.uniforms.density!.value = value;
    this.leather.opacity = value;
    this.leather.transparent = value < 1;
  }
}
