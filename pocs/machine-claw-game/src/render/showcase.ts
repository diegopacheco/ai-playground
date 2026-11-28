import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import type { CollectionEntry } from "../game/collection";
import { fitText, paintTexture } from "./canvasTexture";
import { createPlush } from "./plushMesh";

export type Showcase = { dispose(): void };

const CELL_X = 1.35;
const CELL_Y = 1.6;
const PASTELS = [0xffd6e8, 0xd7ecff, 0xdff7d8, 0xfff0c7, 0xeadcff, 0xffe1d1] as const;
const PEDESTAL = new THREE.CylinderGeometry(0.46, 0.5, 0.14, 40);

export function gridColumns(count: number, aspect: number): number {
  return Math.max(1, Math.min(count, Math.round(Math.sqrt(count * aspect * (CELL_Y / CELL_X)))));
}

function label(entry: CollectionEntry): THREE.Sprite {
  const title = entry.name.toUpperCase();
  const texture = paintTexture(320, 112, (context) => {
    context.fillStyle = "rgba(255, 255, 255, 0.92)";
    context.beginPath();
    context.roundRect(6, 6, 308, 100, 50);
    context.fill();
    context.fillStyle = "#6b3fa0";
    fitText(context, title, "Arial Black, system-ui, sans-serif", 40, 200);
    context.textAlign = "left";
    context.textBaseline = "middle";
    context.fillText(title, 30, 58);
    context.fillStyle = "#ff5c93";
    context.font = "900 40px Arial Black, system-ui, sans-serif";
    context.textAlign = "right";
    context.fillText(`x${entry.count}`, 292, 58);
  });
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, toneMapped: false }));
  sprite.scale.set(1.15, 0.4, 1);
  return sprite;
}

function stand(entry: CollectionEntry, index: number): { readonly cell: THREE.Group; readonly toy: THREE.Group } {
  const cell = new THREE.Group();
  const color = PASTELS[index % PASTELS.length] ?? 0xffffff;
  const pedestal = new THREE.Mesh(PEDESTAL, new THREE.MeshStandardMaterial({ color, roughness: 0.55 }));
  pedestal.position.y = -0.47;
  const toy = createPlush(entry.look, 0.36);
  const tag = label(entry);
  tag.position.y = -0.74;
  cell.add(pedestal, toy, tag);
  return { cell, toy };
}

export function createShowcase(canvas: HTMLCanvasElement, entries: readonly CollectionEntry[]): Showcase {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  const environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environment = environment;
  scene.environmentIntensity = 0.5;
  const key = new THREE.DirectionalLight(0xfff2e6, 1.6);
  key.position.set(2, 4, 5);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xffc9e3, 1.4), key);

  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  const shelf = new THREE.Group();
  scene.add(shelf);
  const stands = entries.map(stand);
  for (const { cell } of stands) shelf.add(cell);

  let spin = 0;
  let spinSpeed = 0;
  let tilt = 0;
  let dragging: { readonly id: number; x: number; y: number } | undefined;
  let last = performance.now();
  let frameId = 0;

  function layout(): void {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    renderer.setSize(width, height, false);
    const aspect = width / Math.max(1, height);
    camera.aspect = aspect;
    const columns = gridColumns(stands.length, aspect);
    const rows = Math.ceil(stands.length / columns);
    stands.forEach(({ cell }, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      const inRow = Math.min(columns, stands.length - row * columns);
      cell.position.set((column - (inRow - 1) / 2) * CELL_X, ((rows - 1) / 2 - row) * CELL_Y + 0.12, 0);
    });
    const half = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const distance = Math.max((rows * CELL_Y) / 2 / half, (columns * CELL_X) / 2 / (half * aspect)) * 1.08 + 0.6;
    camera.position.set(0, 0.35, distance);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }

  function frame(now: number): void {
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    if (!dragging) {
      spinSpeed *= 1 - Math.min(1, dt * 2.5);
      spin += spinSpeed * dt;
      tilt *= 1 - Math.min(1, dt * 2);
    }
    stands.forEach(({ toy }, index) => {
      toy.rotation.set(tilt, spin + Math.sin(now / 1400 + index * 0.7) * 0.45, 0);
      toy.position.y = Math.sin(now / 600 + index) * 0.025;
    });
    renderer.render(scene, camera);
    frameId = requestAnimationFrame(frame);
  }

  function down(event: PointerEvent): void {
    canvas.setPointerCapture(event.pointerId);
    dragging = { id: event.pointerId, x: event.clientX, y: event.clientY };
  }

  function move(event: PointerEvent): void {
    if (!dragging || dragging.id !== event.pointerId) return;
    const dx = event.clientX - dragging.x;
    const dy = event.clientY - dragging.y;
    dragging.x = event.clientX;
    dragging.y = event.clientY;
    spin += dx * 0.012;
    spinSpeed = dx * 0.6;
    tilt = THREE.MathUtils.clamp(tilt + dy * 0.008, -0.6, 0.6);
  }

  function up(event: PointerEvent): void {
    if (dragging?.id === event.pointerId) dragging = undefined;
  }

  const observer = new ResizeObserver(layout);
  observer.observe(canvas);
  canvas.addEventListener("pointerdown", down);
  canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  layout();
  frameId = requestAnimationFrame(frame);

  return {
    dispose: () => {
      cancelAnimationFrame(frameId);
      observer.disconnect();
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("pointercancel", up);
      shelf.traverse((node) => {
        if (node instanceof THREE.Sprite) {
          node.material.map?.dispose();
          node.material.dispose();
        }
        if (node instanceof THREE.Mesh && node.geometry === PEDESTAL) node.material.dispose();
      });
      environment.dispose();
      pmrem.dispose();
      renderer.dispose();
    },
  };
}
