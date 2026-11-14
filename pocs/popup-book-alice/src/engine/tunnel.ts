import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  Group,
  Mesh,
  MeshDepthMaterial,
  MeshStandardMaterial,
  type Object3D,
  PlaneGeometry,
  RGBADepthPacking,
  Vector3,
} from "three";
import { toTexture } from "../art/canvas";
import { accordionZ, easeOutBack, layerFade, SET_DEPTH } from "./timeline";

export const BOOK_W = 10;
export const BOOK_H = 6.4;
const PLEATS = 14;
const PLEAT_AMP = 0.22;

export type Updater = (time: number, dt: number) => void;

interface Item {
  node: Object3D;
  baseZ: number;
}

export type Paper = Mesh<PlaneGeometry, MeshStandardMaterial>;

export function paperMesh(canvas: HTMLCanvasElement, width: number, height: number, anchorBottom = false): Paper {
  const geometry = new PlaneGeometry(width, height);
  if (anchorBottom) geometry.translate(0, height / 2, 0);
  const map = toTexture(canvas);
  const material = new MeshStandardMaterial({ map, alphaTest: 0.5, side: DoubleSide, roughness: 0.92, metalness: 0, transparent: true });
  const mesh = new Mesh(geometry, material);
  mesh.customDepthMaterial = new MeshDepthMaterial({ map, alphaTest: 0.5, depthPacking: RGBADepthPacking });
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function pleatGeometry(side: number): BufferGeometry {
  const positions: number[] = [];
  const pts: [number, number][] = [];
  for (let i = 0; i <= PLEATS; i++) pts.push([side * (BOOK_W / 2 + (i % 2 === 0 ? 0 : PLEAT_AMP)), (i / PLEATS) * SET_DEPTH]);
  for (let i = 0; i < PLEATS; i++) {
    const [x0, z0] = pts[i]!;
    const [x1, z1] = pts[i + 1]!;
    positions.push(x0, 0, z0, x1, 0, z1, x1, BOOK_H, z1, x0, 0, z0, x1, BOOK_H, z1, x0, BOOK_H, z0);
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(new Float32Array(positions), 3));
  geometry.computeVertexNormals();
  return geometry;
}

export class Tunnel {
  readonly group = new Group();
  readonly set = new Group();
  readonly pivot = new Vector3(0, 0, -3);
  openness = 0;
  clock = 0;
  private readonly items: Item[] = [];
  private readonly papers: Paper[] = [];
  private readonly updaters: Updater[] = [];
  private readonly pleats = new Group();

  constructor(z: number, pleatColor: string | null) {
    this.group.position.z = z;
    this.group.add(this.set);
    const material = new MeshStandardMaterial({ color: pleatColor ?? "#ffffff", roughness: 0.85, side: DoubleSide, flatShading: true });
    for (const side of pleatColor ? [-1, 1] : []) {
      const pleat = new Mesh(pleatGeometry(side), material);
      pleat.receiveShadow = true;
      pleat.castShadow = true;
      this.pleats.add(pleat);
    }
    this.pleats.position.set(0, -BOOK_H / 2, -SET_DEPTH);
    this.group.add(this.pleats);
  }

  layer(canvas: HTMLCanvasElement, z: number, inSet = true): Paper {
    const mesh = paperMesh(canvas, BOOK_W, BOOK_H);
    return this.track(mesh, z, inSet);
  }

  cutout(canvas: HTMLCanvasElement, height: number, x: number, y: number, z: number, inSet = true): Group {
    const mesh = paperMesh(canvas, (canvas.width / canvas.height) * height, height, true);
    const holder = new Group();
    holder.position.set(x, y, 0);
    holder.add(mesh);
    this.papers.push(mesh);
    const node = new Group();
    node.add(holder);
    this.register(node, z, inSet);
    return holder;
  }

  solid<T extends Object3D>(object: T, x: number, y: number, z: number, inSet = true): T {
    object.traverse((child) => {
      child.castShadow = true;
      child.receiveShadow = true;
    });
    const node = new Group();
    object.position.x += x;
    object.position.y += y;
    node.add(object);
    this.register(node, z, inSet);
    return object;
  }

  free(object: Object3D): void {
    this.group.add(object);
  }

  onUpdate(updater: Updater): void {
    this.updaters.push(updater);
  }

  scaleSet(scale: number): void {
    this.set.scale.set(scale, scale, 1);
    this.set.position.set(this.pivot.x * (1 - scale), this.pivot.y * (1 - scale), 0);
  }

  setOpenness(openness: number): void {
    if (openness <= 0.001) this.clock = 0;
    this.openness = openness;
    for (const item of this.items) item.node.position.z = accordionZ(item.baseZ, openness);
    this.pleats.scale.z = Math.max(easeOutBack(Math.min(Math.max(openness, 0), 1)), 0.02);
  }

  fade(cameraZ: number): void {
    const world = new Vector3();
    for (const paper of this.papers) {
      paper.getWorldPosition(world);
      const opacity = layerFade(cameraZ - world.z) * ((paper.userData.alpha as number | undefined) ?? 1);
      paper.material.opacity = opacity;
      paper.visible = opacity > 0.01;
    }
  }

  update(dt: number): void {
    this.clock += dt;
    for (const updater of this.updaters) updater(this.clock, dt);
  }

  dispose(): void {
    this.group.traverse((child) => {
      if (child instanceof Mesh) {
        child.geometry.dispose();
        const material = child.material as MeshStandardMaterial;
        material.map?.dispose();
        material.dispose();
      }
    });
  }

  private track(mesh: Paper, z: number, inSet: boolean): Paper {
    this.papers.push(mesh);
    const node = new Group();
    node.add(mesh);
    this.register(node, z, inSet);
    return mesh;
  }

  private register(node: Group, z: number, inSet: boolean): void {
    (inSet ? this.set : this.group).add(node);
    this.items.push({ node, baseZ: z });
  }
}
