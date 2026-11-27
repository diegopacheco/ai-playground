import * as THREE from "three";
import { PALETTE } from "./palette";

export type OutlinedBoxOptions = {
  readonly width: number;
  readonly height: number;
  readonly depth: number;
  readonly color?: number;
  readonly material?: THREE.Material | THREE.Material[];
  readonly outline?: boolean;
  readonly castShadow?: boolean;
  readonly receiveShadow?: boolean;
};

const sharedLineMaterial = new THREE.LineBasicMaterial({
  color: PALETTE.ink,
  transparent: true,
  opacity: 0.88,
});

export function setOutlineVisibility(visible: boolean): void {
  sharedLineMaterial.visible = visible;
}

export function addInkEdges(
  mesh: THREE.Mesh,
  thresholdAngle = 20,
): THREE.LineSegments {
  const edges = new THREE.EdgesGeometry(mesh.geometry, thresholdAngle);
  const line = new THREE.LineSegments(edges, sharedLineMaterial);
  line.renderOrder = 2;
  mesh.add(line);
  return line;
}

export function createOutlinedBox(opts: OutlinedBoxOptions): THREE.Mesh {
  const geo = new THREE.BoxGeometry(opts.width, opts.height, opts.depth);
  const mat =
    opts.material ??
    new THREE.MeshLambertMaterial({ color: opts.color ?? PALETTE.sandstone });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = opts.castShadow ?? true;
  mesh.receiveShadow = opts.receiveShadow ?? true;
  if (opts.outline !== false) {
    addInkEdges(mesh);
  }
  return mesh;
}

export function createOutlinedCylinder(
  radiusTop: number,
  radiusBottom: number,
  height: number,
  radialSegments: number,
  color: number,
  outline = true,
): THREE.Mesh {
  const geo = new THREE.CylinderGeometry(
    radiusTop,
    radiusBottom,
    height,
    radialSegments,
  );
  const mat = new THREE.MeshLambertMaterial({ color });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  if (outline) {
    addInkEdges(mesh, 26);
  }
  return mesh;
}

export function createGableRoof(
  width: number,
  height: number,
  depth: number,
  color: number,
): THREE.Mesh {
  const shape = new THREE.Shape();
  shape.moveTo(-width / 2, 0);
  shape.lineTo(width / 2, 0);
  shape.lineTo(0, height);
  shape.closePath();

  const geo = new THREE.ExtrudeGeometry(shape, {
    depth,
    bevelEnabled: false,
  });
  geo.translate(0, 0, -depth / 2);

  const mat = new THREE.MeshLambertMaterial({ color });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  addInkEdges(mesh, 18);
  return mesh;
}

export function createFacadeMaterials(
  sideTexture: THREE.CanvasTexture,
  roofColor: number,
  bottomColor = PALETTE.ink,
): THREE.MeshLambertMaterial[] {
  const sideMat = new THREE.MeshLambertMaterial({ map: sideTexture });
  const topMat = new THREE.MeshLambertMaterial({ color: roofColor });
  const bottomMat = new THREE.MeshLambertMaterial({ color: bottomColor });
  return [sideMat, sideMat, topMat, bottomMat, sideMat, sideMat];
}
