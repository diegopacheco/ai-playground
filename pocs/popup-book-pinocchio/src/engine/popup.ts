import { CanvasTexture, DoubleSide, Group, Mesh, MeshStandardMaterial, type Object3D, PlaneGeometry, Vector3 } from "three";
import { toTexture } from "../art/canvas";
import { easeOutBack, foldAngle, pieceProgress } from "./timeline";

export const FLOOR_Y = 0.6;
export const STAGE_W = 8.6;
export const STAGE_D = 4.4;

type PieceKind = "fold" | "grow";

interface Piece {
  hinge: Group;
  delay: number;
  kind: PieceKind;
  layer: number;
}

export interface CutoutOptions {
  delay?: number;
  turn?: number;
  flip?: boolean;
}

export interface Mood {
  key: string;
  fill: string;
  ambient: number;
}

export type Updater = (time: number, dt: number) => void;

export type Cutout = Mesh<PlaneGeometry, MeshStandardMaterial>;

export function cutoutMesh(canvas: HTMLCanvasElement, height: number, flip = false): Cutout {
  const width = (canvas.width / canvas.height) * height;
  const geometry = new PlaneGeometry(width, height);
  geometry.translate(0, height / 2, 0);
  const material = new MeshStandardMaterial({ map: toTexture(canvas), alphaTest: 0.5, side: DoubleSide, roughness: 0.82, metalness: 0 });
  const mesh = new Mesh(geometry, material);
  if (flip) mesh.scale.x = -1;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

export class PopupScene {
  readonly group = new Group();
  readonly backdrop: CanvasTexture;
  readonly focus = new Vector3(0, 1.6, 0);
  mood: Mood = { key: "#ffe2b0", fill: "#ffd08a", ambient: 0.5 };
  openness = 0;
  private readonly pieces: Piece[] = [];
  private readonly updaters: Updater[] = [];

  constructor(backdrop: HTMLCanvasElement) {
    this.backdrop = toTexture(backdrop);
  }

  cutout(canvas: HTMLCanvasElement, height: number, x: number, z: number, options: CutoutOptions = {}): Cutout {
    const mesh = cutoutMesh(canvas, height, options.flip);
    const hinge = this.hinge(x, z, options.turn ?? 0, mesh);
    this.pieces.push({ hinge, delay: options.delay ?? this.autoDelay(z), kind: "fold", layer: this.pieces.length });
    return mesh;
  }

  solid<T extends Object3D>(object: T, x: number, z: number, delay?: number, turn = 0): T {
    object.traverse((child) => {
      child.castShadow = true;
      child.receiveShadow = true;
    });
    const hinge = this.hinge(x, z, turn, object);
    this.pieces.push({ hinge, delay: delay ?? this.autoDelay(z), kind: "grow", layer: this.pieces.length });
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
    this.group.visible = openness > 0.001;
    for (const piece of this.pieces) {
      if (piece.kind === "fold") {
        piece.hinge.rotation.x = -foldAngle(openness, piece.delay);
        piece.hinge.position.y = FLOOR_Y + 0.004 + piece.layer * 0.002;
      } else {
        const p = pieceProgress(openness, piece.delay);
        piece.hinge.visible = p > 0.001;
        piece.hinge.scale.set(0.7 + 0.3 * p, Math.max(easeOutBack(p), 0.001), 0.7 + 0.3 * p);
        piece.hinge.rotation.y = (1 - p) * 0.9;
      }
    }
  }

  update(time: number, dt: number): void {
    for (const updater of this.updaters) updater(time, dt);
  }

  dispose(): void {
    this.backdrop.dispose();
    this.group.traverse((child) => {
      if (child instanceof Mesh) {
        child.geometry.dispose();
        const material = child.material as MeshStandardMaterial;
        material.map?.dispose();
        material.dispose();
      }
    });
  }

  private hinge(x: number, z: number, turn: number, content: Object3D): Group {
    const hinge = new Group();
    hinge.position.set(x, FLOOR_Y, z);
    const yaw = new Group();
    yaw.rotation.y = turn;
    yaw.add(content);
    hinge.add(yaw);
    this.group.add(hinge);
    return hinge;
  }

  private autoDelay(z: number): number {
    return Math.min(Math.max((z + STAGE_D / 2) / STAGE_D, 0), 1) * 0.45;
  }
}
