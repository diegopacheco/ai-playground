import * as THREE from "three";

export type CameraRig = {
  fit(aspect: number): void;
  toggle(): void;
  orbit(dx: number, dy: number): void;
  zoom(scale: number): void;
  update(dt: number): void;
};

type View = { readonly yaw: number; readonly pitch: number; readonly zoom: number };

const FRONT: View = { yaw: 0, pitch: 0.1, zoom: 1 };
const SIDE: View = { yaw: 1.15, pitch: 0.22, zoom: 1 };
const LIMITS = { yaw: 1.9, pitchMin: 0.02, pitchMax: 1.25, zoomMin: 0.45, zoomMax: 1.3 } as const;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function createCameraRig(camera: THREE.PerspectiveCamera): CameraRig {
  const target = new THREE.Vector3(0, 0.6, 0);
  let distance = 4;
  let wanted: View = FRONT;
  let now: View = FRONT;

  function fit(aspect: number): void {
    const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
    distance = Math.max(1.42, 0.8 / aspect) / Math.tan(halfFov);
    target.y = aspect < 1 ? 0.42 : 0.6;
  }

  function update(dt: number): void {
    const ease = Math.min(1, dt * 7);
    now = {
      yaw: now.yaw + (wanted.yaw - now.yaw) * ease,
      pitch: now.pitch + (wanted.pitch - now.pitch) * ease,
      zoom: now.zoom + (wanted.zoom - now.zoom) * ease,
    };
    const reach = distance * now.zoom;
    const flat = Math.cos(now.pitch) * reach;
    camera.position.set(
      target.x + Math.sin(now.yaw) * flat,
      target.y + Math.sin(now.pitch) * reach,
      target.z + Math.cos(now.yaw) * flat,
    );
    camera.lookAt(target);
  }

  return {
    fit,
    toggle: () => {
      wanted = Math.abs(wanted.yaw) < 0.3 ? SIDE : FRONT;
    },
    orbit: (dx, dy) => {
      wanted = {
        ...wanted,
        yaw: clamp(wanted.yaw - dx * 0.006, -LIMITS.yaw, LIMITS.yaw),
        pitch: clamp(wanted.pitch + dy * 0.005, LIMITS.pitchMin, LIMITS.pitchMax),
      };
    },
    zoom: (scale) => {
      wanted = { ...wanted, zoom: clamp(wanted.zoom * scale, LIMITS.zoomMin, LIMITS.zoomMax) };
    },
    update,
  };
}
