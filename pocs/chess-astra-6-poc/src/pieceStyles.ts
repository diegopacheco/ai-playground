import * as THREE from 'three';
import { canvasTexture } from './parts';

export const pieceStyles = { classic: 'Classic', marble: 'Marble', wood: 'Wood', steel: 'Steel', glass: 'Glass', plastic: 'Plastic', stone: 'Stone' } as const;
export type PieceStyle = keyof typeof pieceStyles;

const painters = {
  marble(ctx: CanvasRenderingContext2D) {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, 256, 256);
    for (let vein = 0; vein < 22; vein++) {
      ctx.strokeStyle = `rgba(70, 72, 84, ${0.35 + Math.random() * 0.5})`;
      ctx.lineWidth = 1.5 + Math.random() * 4;
      ctx.beginPath();
      let x = Math.random() * 256;
      ctx.moveTo(x, 0);
      for (let y = 16; y <= 256; y += 16) {
        x += (Math.random() - 0.5) * 36;
        ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  },
  wood(ctx: CanvasRenderingContext2D) {
    for (let x = 0; x < 256; x++) {
      const shade = 0.68 + Math.sin(x * 0.3 + Math.sin(x * 0.05) * 4) * 0.24 + Math.random() * 0.08;
      ctx.fillStyle = `rgb(${255 * shade}, ${232 * shade}, ${205 * shade})`;
      ctx.fillRect(x, 0, 1, 256);
    }
  },
  stone(ctx: CanvasRenderingContext2D) {
    ctx.fillStyle = '#d6d6d2';
    ctx.fillRect(0, 0, 256, 256);
    for (let i = 0; i < 5000; i++) {
      const tone = 90 + Math.random() * 165;
      ctx.fillStyle = `rgb(${tone}, ${tone}, ${tone - 6})`;
      ctx.fillRect(Math.random() * 256, Math.random() * 256, 1 + Math.random() * 2, 1 + Math.random() * 2);
    }
  },
};
const maps = new Map<keyof typeof painters, THREE.Texture>();

function map(name: keyof typeof painters) {
  if (!maps.has(name)) {
    const texture = canvasTexture(256, 256, painters[name]);
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(2, 1);
    maps.set(name, texture);
  }
  return maps.get(name)!;
}

const looks: Record<PieceStyle, () => THREE.MeshPhysicalMaterialParameters> = {
  classic: () => ({ metalness: 0.25, roughness: 0.32, clearcoat: 0.6, clearcoatRoughness: 0.2, envMapIntensity: 0.22 }),
  marble: () => ({ map: map('marble'), roughness: 0.14, clearcoat: 0.8, envMapIntensity: 0.6 }),
  wood: () => ({ map: map('wood'), roughness: 0.62, clearcoat: 0.2, envMapIntensity: 0.2 }),
  steel: () => ({ metalness: 1, roughness: 0.22, envMapIntensity: 1.3 }),
  glass: () => ({ roughness: 0.04, transmission: 1, thickness: 0.8, clearcoat: 1, envMapIntensity: 1.2 }),
  plastic: () => ({ roughness: 0.36, clearcoat: 0.5, clearcoatRoughness: 0.25, envMapIntensity: 0.35 }),
  stone: () => ({ map: map('stone'), roughness: 0.95, envMapIntensity: 0.1 }),
};

export function applyPieceStyle(material: THREE.MeshPhysicalMaterial, style: PieceStyle) {
  material.setValues({ map: null, metalness: 0, roughness: 0.5, clearcoat: 0, clearcoatRoughness: 0.1, transmission: 0, thickness: 0, ior: 1.5, envMapIntensity: 0, ...looks[style]() });
  material.needsUpdate = true;
}
