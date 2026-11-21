import * as THREE from "three";

export type CameraRig = {
  fit(aspect: number): void;
  toggle(): void;
  update(dt: number, clawX: number, clawZ: number): void;
};

const SIDE_ANGLE = 1.15;

export function createCameraRig(camera: THREE.PerspectiveCamera): CameraRig {
  const target = new THREE.Vector3(0, 0.62, 0);
  let distance = 3.4;
  let wanted = 0;
  let angle = 0;
  let swayX = 0;
  let swayZ = 0;

  function fit(aspect: number): void {
    const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
    distance = Math.max(1.42, 0.8 / aspect) / Math.tan(halfFov);
    target.y = aspect < 1 ? 0.42 : 0.6;
  }

  function update(dt: number, clawX: number, clawZ: number): void {
    const ease = Math.min(1, dt * 4);
    angle += (wanted - angle) * ease;
    swayX += (clawX * 0.25 - swayX) * ease;
    swayZ += (clawZ * 0.12 - swayZ) * ease;
    camera.position.set(Math.sin(angle) * distance + swayX, target.y + 0.4 - swayZ, Math.cos(angle) * distance);
    camera.lookAt(target);
  }

  return {
    fit,
    toggle: () => {
      wanted = wanted === 0 ? SIDE_ANGLE : 0;
    },
    update,
  };
}
