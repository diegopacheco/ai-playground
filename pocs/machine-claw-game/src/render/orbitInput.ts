import type { CameraRig } from "./cameraRig";

type Point = { readonly x: number; readonly y: number };

function spread(points: readonly Point[]): number {
  const [a, b] = points;
  return a && b ? Math.hypot(a.x - b.x, a.y - b.y) : 0;
}

export function attachOrbitInput(canvas: HTMLCanvasElement, rig: CameraRig): () => void {
  const pointers = new Map<number, Point>();

  function down(event: PointerEvent): void {
    canvas.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  }

  function move(event: PointerEvent): void {
    const before = pointers.get(event.pointerId);
    if (!before) return;
    const apart = spread([...pointers.values()]);
    const here = { x: event.clientX, y: event.clientY };
    pointers.set(event.pointerId, here);
    if (pointers.size === 1) {
      rig.orbit(here.x - before.x, here.y - before.y);
      return;
    }
    const apartNow = spread([...pointers.values()]);
    if (apart > 0 && apartNow > 0) rig.zoom(apart / apartNow);
  }

  function up(event: PointerEvent): void {
    pointers.delete(event.pointerId);
  }

  function wheel(event: WheelEvent): void {
    event.preventDefault();
    rig.zoom(Math.exp(event.deltaY * 0.0012));
  }

  canvas.addEventListener("pointerdown", down);
  canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("wheel", wheel, { passive: false });
  return () => {
    canvas.removeEventListener("pointerdown", down);
    canvas.removeEventListener("pointermove", move);
    canvas.removeEventListener("pointerup", up);
    canvas.removeEventListener("pointercancel", up);
    canvas.removeEventListener("wheel", wheel);
  };
}
