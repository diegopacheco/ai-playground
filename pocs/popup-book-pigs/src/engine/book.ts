import {
  BackSide,
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  type CanvasTexture,
  CylinderGeometry,
  FrontSide,
  Group,
  Mesh,
  MeshStandardMaterial,
  PlaneGeometry,
  type Texture,
} from "three";
import { linear, paint, toTexture } from "../art/canvas";
import { PAGE_D, PAGE_W, PAGE_Y } from "./popup";

const SEGMENTS = 48;

export type Half = "left" | "right" | "mirroredLeft";

export function halfTexture(source: CanvasTexture, half: Half): Texture {
  const texture = source.clone();
  texture.repeat.set(half === "mirroredLeft" ? -0.5 : 0.5, 1);
  texture.offset.set(half === "left" ? 0 : 0.5, 0);
  texture.needsUpdate = true;
  return texture;
}

function leatherTexture(): CanvasTexture {
  return toTexture(
    paint(512, 512, (ctx, w, h) => {
      ctx.fillStyle = linear(ctx, 0, h, ["#8e2234", "#5e1222"]);
      ctx.fillRect(0, 0, w, h);
      for (let i = 0; i < 3000; i++) {
        ctx.fillStyle = `rgba(0,0,0,${Math.random() * 0.12})`;
        ctx.fillRect(Math.random() * w, Math.random() * h, 3, 3);
      }
      ctx.strokeStyle = "#e3b955";
      ctx.lineWidth = 8;
      ctx.strokeRect(22, 22, w - 44, h - 44);
      ctx.lineWidth = 3;
      ctx.strokeRect(36, 36, w - 72, h - 72);
    }),
  );
}

function pageEdgeTexture(): CanvasTexture {
  return toTexture(
    paint(64, 256, (ctx, w, h) => {
      ctx.fillStyle = "#f4e6c4";
      ctx.fillRect(0, 0, w, h);
      for (let y = 0; y < h; y += 4) {
        ctx.fillStyle = y % 8 === 0 ? "#d9c49a" : "#fbf2dc";
        ctx.fillRect(0, y, w, 2);
      }
    }),
  );
}

class Leaf {
  readonly group = new Group();
  private readonly geometry = new BufferGeometry();
  private readonly front: MeshStandardMaterial;
  private readonly back: MeshStandardMaterial;
  private readonly positions: Float32Array;

  constructor() {
    const count = (SEGMENTS + 1) * 2;
    this.positions = new Float32Array(count * 3);
    const uvs = new Float32Array(count * 2);
    const indices: number[] = [];
    for (let i = 0; i <= SEGMENTS; i++) {
      for (let j = 0; j <= 1; j++) {
        const k = i * 2 + j;
        uvs[k * 2] = i / SEGMENTS;
        uvs[k * 2 + 1] = 1 - j;
      }
      if (i < SEGMENTS) {
        const a = i * 2;
        indices.push(a, a + 1, a + 2, a + 2, a + 1, a + 3);
      }
    }
    this.geometry.setAttribute("position", new BufferAttribute(this.positions, 3));
    this.geometry.setAttribute("uv", new BufferAttribute(uvs, 2));
    this.geometry.setIndex(indices);
    this.front = new MeshStandardMaterial({ side: FrontSide, roughness: 0.9 });
    this.back = new MeshStandardMaterial({ side: BackSide, roughness: 0.9 });
    const frontMesh = new Mesh(this.geometry, this.front);
    const backMesh = new Mesh(this.geometry, this.back);
    for (const mesh of [frontMesh, backMesh]) {
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.group.add(mesh);
    }
    this.group.visible = false;
    this.bend(0);
  }

  setFaces(front: Texture, back: Texture): void {
    this.front.map = front;
    this.back.map = back;
    this.front.needsUpdate = true;
    this.back.needsUpdate = true;
  }

  bend(progress: number): void {
    const theta = progress * Math.PI;
    const curl = Math.sin(progress * Math.PI) * 0.85;
    const lift = Math.sin(progress * Math.PI) * 0.08;
    const ds = PAGE_W / SEGMENTS;
    let x = 0;
    let y = PAGE_Y + 0.014 + lift;
    for (let i = 0; i <= SEGMENTS; i++) {
      for (let j = 0; j <= 1; j++) {
        const k = (i * 2 + j) * 3;
        this.positions[k] = x;
        this.positions[k + 1] = y;
        this.positions[k + 2] = -PAGE_D / 2 + j * PAGE_D;
      }
      const angle = theta - curl * (i / SEGMENTS);
      x += Math.cos(angle) * ds;
      y += Math.sin(angle) * ds;
    }
    this.geometry.attributes.position!.needsUpdate = true;
    this.geometry.computeVertexNormals();
    this.geometry.computeBoundingSphere();
  }
}

export class Book {
  readonly group = new Group();
  readonly leaf = new Leaf();
  private readonly left: MeshStandardMaterial;
  private readonly right: MeshStandardMaterial;

  constructor() {
    const leather = new MeshStandardMaterial({ map: leatherTexture(), roughness: 0.55, metalness: 0.05 });
    const edge = new MeshStandardMaterial({ map: pageEdgeTexture(), roughness: 0.95 });
    const paper = new MeshStandardMaterial({ color: "#f7ecd3", roughness: 0.95 });
    for (const side of [-1, 1]) {
      const cover = new Mesh(new BoxGeometry(PAGE_W + 0.28, 0.12, PAGE_D + 0.36), leather);
      cover.position.set(side * (PAGE_W / 2 + 0.1), 0.06, 0);
      const stack = new Mesh(new BoxGeometry(PAGE_W, PAGE_Y - 0.12, PAGE_D), [edge, edge, paper, paper, edge, edge]);
      stack.position.set(side * PAGE_W / 2, 0.12 + (PAGE_Y - 0.12) / 2, 0);
      for (const mesh of [cover, stack]) {
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        this.group.add(mesh);
      }
    }
    const spine = new Mesh(new CylinderGeometry(0.22, 0.22, PAGE_D + 0.36, 24, 1, false, Math.PI / 2, Math.PI), leather);
    spine.rotation.x = Math.PI / 2;
    spine.position.y = 0.12;
    spine.castShadow = true;
    this.group.add(spine);
    this.left = new MeshStandardMaterial({ roughness: 0.92 });
    this.right = new MeshStandardMaterial({ roughness: 0.92 });
    for (const [material, x] of [[this.left, -PAGE_W / 2], [this.right, PAGE_W / 2]] as const) {
      const page = new Mesh(new PlaneGeometry(PAGE_W, PAGE_D), material);
      page.rotation.x = -Math.PI / 2;
      page.position.set(x, PAGE_Y + 0.002, 0);
      page.receiveShadow = true;
      this.group.add(page);
    }
    this.group.add(this.leaf.group);
  }

  setLeft(texture: Texture): void {
    this.left.map = texture;
    this.left.needsUpdate = true;
  }

  setRight(texture: Texture): void {
    this.right.map = texture;
    this.right.needsUpdate = true;
  }
}
