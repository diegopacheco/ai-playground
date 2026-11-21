import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

export type Stage = {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  resize(): number;
  render(): void;
  dispose(): void;
};

export function createStage(canvas: HTMLCanvasElement): Stage {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.92;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x120a24);
  scene.fog = new THREE.Fog(0x120a24, 10, 30);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environment = environment;
  scene.environmentIntensity = 0.4;

  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 40);

  function resize(): number {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(1, height);
    camera.updateProjectionMatrix();
    return camera.aspect;
  }

  return {
    scene,
    camera,
    resize,
    render: () => renderer.render(scene, camera),
    dispose: () => {
      environment.dispose();
      pmrem.dispose();
      renderer.dispose();
    },
  };
}
