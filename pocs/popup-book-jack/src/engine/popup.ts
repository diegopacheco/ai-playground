import { type CanvasTexture, DoubleSide, Group, Mesh, MeshBasicMaterial, MeshStandardMaterial, type Object3D, PlaneGeometry } from "three";
import { paint, toTexture } from "../art/canvas";
import { easeOutBack, foldAngle, pieceProgress } from "./timeline";

export const PAGE_W = 4.2;
export const PAGE_D = 5.88;
export const PAGE_Y = 0.34;

type PieceKind = "fold" | "grow";

interface Piece {
  hinge: Group;
  delay: number;
  kind: PieceKind;
  layer: number;
}

interface Footing {
  mesh: Mesh;
  shadow: Mesh<PlaneGeometry, MeshBasicMaterial>;
  tab: Mesh;
  delay: number;
}

let shadowCache: CanvasTexture | undefined;
function contactTexture(): CanvasTexture {
  shadowCache ??= toTexture(
    paint(128, 64, (ctx, w, h) => {
      const g = ctx.createRadialGradient(w / 2, h / 2, 2, w / 2, h / 2, w / 2);
      g.addColorStop(0, "rgba(40,25,10,0.85)");
      g.addColorStop(0.5, "rgba(40,25,10,0.35)");
      g.addColorStop(1, "rgba(40,25,10,0)");
      ctx.fillStyle = g;
      ctx.scale(1, h / w);
      ctx.fillRect(0, 0, w, w);
    }),
  );
  return shadowCache;
}

const tabMaterial = new MeshStandardMaterial({ color: "#e2d2ac", roughness: 0.95 });

export interface CutoutOptions {
  delay?: number;
  turn?: number;
  flip?: boolean;
}

export type Updater = (time: number, dt: number) => void;

export function cutoutMesh(canvas: HTMLCanvasElement, height: number, flip = false): Mesh<PlaneGeometry, MeshStandardMaterial> {
  const width = (canvas.width / canvas.height) * height;
  const geometry = new PlaneGeometry(width, height);
  geometry.translate(0, height / 2, 0);
  const material = new MeshStandardMaterial({
    map: toTexture(canvas),
    alphaTest: 0.4,
    alphaToCoverage: true,
    side: DoubleSide,
    roughness: 0.82,
    metalness: 0,
  });
  const mesh = new Mesh(geometry, material);
  if (flip) mesh.scale.x = -1;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

export class PopupScene {
  readonly group = new Group();
  readonly backdrop: CanvasTexture;
  lookHeight = 1.0;
  since = 0;
  shake = 0;
  openness = 0;
  private readonly pieces: Piece[] = [];
  private readonly updaters: Updater[] = [];
  private readonly footings: Footing[] = [];

  constructor(backdrop: HTMLCanvasElement) {
    this.backdrop = toTexture(backdrop);
  }

  cutout(canvas: HTMLCanvasElement, height: number, x: number, z: number, options: CutoutOptions = {}): Mesh<PlaneGeometry, MeshStandardMaterial> {
    const mesh = cutoutMesh(canvas, height, options.flip);
    const hinge = this.hinge(x, z, options.turn ?? 0, mesh);
    const delay = options.delay ?? this.autoDelay(z);
    this.pieces.push({ hinge, delay, kind: "fold", layer: this.pieces.length });
    this.footing(mesh, x, z, options.turn ?? 0, delay, (canvas.width / canvas.height) * height);
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
    for (const foot of this.footings) foot.tab.visible = pieceProgress(openness, foot.delay) > 0.02;
    for (const piece of this.pieces) {
      if (piece.kind === "fold") {
        piece.hinge.rotation.x = foldAngle(openness, piece.delay);
        const sink = openness < 0.1 ? (0.1 - openness) * 0.8 : 0;
        piece.hinge.position.y = PAGE_Y + 0.004 + piece.layer * 0.0025 - sink;
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
    for (const foot of this.footings) {
      const lift = Math.max(foot.mesh.position.y, 0);
      const grounded = lift < 0.45 && foot.mesh.visible;
      const stand = pieceProgress(this.openness, foot.delay);
      const spread = 1 / (1 + lift * 1.8);
      foot.shadow.visible = grounded;
      foot.tab.visible = grounded && stand > 0.02;
      foot.shadow.scale.set(spread, spread, 1);
      foot.shadow.material.opacity = 0.75 * stand * spread;
    }
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

  private footing(mesh: Mesh, x: number, z: number, turn: number, delay: number, width: number): void {
    const base = new Group();
    base.position.set(x, PAGE_Y, z);
    base.rotation.y = turn;
    const shadow = new Mesh(new PlaneGeometry(width * 0.95, Math.min(0.5, width * 0.35)), new MeshBasicMaterial({ map: contactTexture(), transparent: true, depthWrite: false }));
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.set(0, 0.006, 0.04);
    const tab = new Mesh(new PlaneGeometry(width * 0.4, 0.11), tabMaterial);
    tab.rotation.x = -Math.PI / 2;
    tab.position.set(0, 0.004, 0.09);
    tab.receiveShadow = true;
    base.add(shadow, tab);
    this.group.add(base);
    this.footings.push({ mesh, shadow, tab, delay });
  }

  private hinge(x: number, z: number, turn: number, content: Object3D): Group {
    const hinge = new Group();
    hinge.position.set(x, PAGE_Y, z);
    const yaw = new Group();
    yaw.rotation.y = turn;
    yaw.add(content);
    hinge.add(yaw);
    this.group.add(hinge);
    return hinge;
  }

  private autoDelay(z: number): number {
    return Math.min(Math.max((z + PAGE_D / 2) / PAGE_D, 0), 1) * 0.45;
  }
}
