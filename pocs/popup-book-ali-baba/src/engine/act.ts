import { Color, CylinderGeometry, Group, Mesh, MeshStandardMaterial, type Object3D, Vector3 } from "three";
import { MASK_LAYER, maskMaterial } from "./materials";
import type { Sil } from "./silhouette";
import { easeOutBack, foldAngle, pieceProgress } from "./timeline";

export const BASE_Y = 0.1;
export const REST_LAMP = new Vector3(0, 2.2, -4.2);

type PieceKind = "fold" | "rise" | "drop" | "grow";

interface Piece {
  hinge: Group;
  delay: number;
  kind: PieceKind;
  drop: number;
  y: number;
}

export interface PlaceOptions {
  y?: number;
  delay?: number;
  kind?: "fold" | "rise" | "drop";
  rod?: boolean;
}

export type Updater = (time: number, dt: number) => void;

const rodGeometry = new CylinderGeometry(0.012, 0.012, 1.6, 6);
rodGeometry.rotateX(Math.PI / 2);
rodGeometry.translate(0, 0, -0.8);
const rodMaterial = new MeshStandardMaterial({ color: "#7a5a32", roughness: 0.7 });

export function castShadow(object: Object3D, density = 1, soft = 0.6): void {
  const meshes: Mesh[] = [];
  object.traverse((child) => {
    if (child instanceof Mesh && child.layers.isEnabled(0)) meshes.push(child);
  });
  for (const mesh of meshes) {
    const source = mesh.material as MeshStandardMaterial;
    const shadow = new Mesh(mesh.geometry, maskMaterial(source.map ?? null, density, soft));
    shadow.layers.set(MASK_LAYER);
    mesh.add(shadow);
  }
}

export class Act {
  readonly group = new Group();
  readonly lamp = REST_LAMP.clone();
  readonly top = new Color("#ffb45c");
  readonly bottom = new Color("#c2410c");
  lampMotion: (time: number, out: Vector3) => void = () => {};
  levelAt: (time: number) => number = () => 1;
  openness = 0;
  clock = 0;
  private readonly pieces: Piece[] = [];
  private readonly updaters: Updater[] = [];

  tint(top: string, bottom: string): this {
    this.top.set(top);
    this.bottom.set(bottom);
    return this;
  }

  place(sil: Sil, x: number, depth: number, options: PlaceOptions = {}): Sil {
    const hinge = new Group();
    const y = options.y ?? 0;
    hinge.position.set(x, BASE_Y + y, -depth);
    hinge.add(sil.root);
    if (options.rod) {
      const rod = new Mesh(rodGeometry, rodMaterial);
      rod.position.set(0, sil.height * 0.55, -0.01);
      sil.root.add(rod);
    }
    this.group.add(hinge);
    this.pieces.push({ hinge, delay: options.delay ?? this.autoDelay(depth), kind: options.kind ?? "fold", drop: sil.height + y + 0.6, y: BASE_Y + y });
    return sil;
  }

  solid<T extends Object3D>(object: T, x: number, depth: number, y = 0, delay?: number): T {
    const hinge = new Group();
    hinge.position.set(x, BASE_Y + y, -depth);
    hinge.add(object);
    this.group.add(hinge);
    this.pieces.push({ hinge, delay: delay ?? this.autoDelay(depth), kind: "grow", drop: 0, y: BASE_Y + y });
    return object;
  }

  free(object: Object3D): void {
    this.group.add(object);
  }

  onUpdate(updater: Updater): void {
    this.updaters.push(updater);
  }

  setOpenness(openness: number): void {
    this.openness = openness;
    if (openness <= 0.001) this.clock = 0;
    this.group.visible = openness > 0.001;
    for (const piece of this.pieces) {
      const p = pieceProgress(openness, piece.delay);
      if (piece.kind === "fold") {
        piece.hinge.rotation.x = -foldAngle(openness, piece.delay);
      } else if (piece.kind === "rise") {
        piece.hinge.position.y = piece.y - (1 - easeOutBack(p)) * piece.drop;
      } else if (piece.kind === "drop") {
        piece.hinge.position.y = piece.y + (1 - easeOutBack(p)) * 3.2;
      } else {
        piece.hinge.visible = p > 0.001;
        piece.hinge.scale.setScalar(Math.max(easeOutBack(p), 0.001));
      }
    }
  }

  update(dt: number): void {
    this.clock += dt;
    for (const updater of this.updaters) updater(this.clock, dt);
  }

  dispose(): void {
    this.group.traverse((child) => {
      if (child instanceof Mesh && child.geometry !== rodGeometry) {
        child.geometry.dispose();
        const material = child.material as MeshStandardMaterial;
        material.map?.dispose();
        material.dispose();
      }
    });
  }

  private autoDelay(depth: number): number {
    return Math.min(Math.max(1 - depth / 2.5, 0), 1) * 0.4;
  }
}
