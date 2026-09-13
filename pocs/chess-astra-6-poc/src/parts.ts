import * as THREE from 'three';

export const gold = new THREE.MeshStandardMaterial({ color: '#b39051', metalness: 0.72, roughness: 0.3 });
export const ivory = new THREE.MeshStandardMaterial({ color: '#f9ecd2', metalness: 0.15, roughness: 0.3 });
export const jade = new THREE.MeshStandardMaterial({ color: '#31574d', metalness: 0.4, roughness: 0.3 });
export const stone = new THREE.MeshStandardMaterial({ color: '#d5cbb9', roughness: 0.9 });

export function mesh(geometry: THREE.BufferGeometry, material: THREE.Material, x = 0, y = 0, z = 0) {
  const object = new THREE.Mesh(geometry, material);
  object.position.set(x, y, z);
  object.castShadow = true;
  object.receiveShadow = true;
  return object;
}

export function canvasTexture(width: number, height: number, draw: (ctx: CanvasRenderingContext2D) => void) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  draw(canvas.getContext('2d')!);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
