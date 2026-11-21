import * as THREE from "three";

export function paintTexture(
  width: number,
  height: number,
  paint: (context: CanvasRenderingContext2D) => void,
): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("2d canvas is not available");
  paint(context);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

export function fitText(context: CanvasRenderingContext2D, text: string, family: string, size: number, width: number): void {
  let current = size;
  context.font = `900 ${current}px ${family}`;
  while (context.measureText(text).width > width && current > 8) {
    current -= 2;
    context.font = `900 ${current}px ${family}`;
  }
}
