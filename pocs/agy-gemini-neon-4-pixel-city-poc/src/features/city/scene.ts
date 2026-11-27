import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CAMERA_PRESETS, findLandmarkById } from "./catalog";
import { buildTerrain } from "./terrain";
import { buildLandmarks } from "./landmarks";
import { buildCityBlocks } from "./buildings";
import { buildFoliageAndProps } from "./foliage";
import { buildTraffic } from "./vehicles";
import { LIGHTING_PRESETS } from "./palette";
import { setOutlineVisibility } from "./geometry";
import { logger } from "../../lib/logger";
import type { CameraPreset, LandmarkId, LightingMode } from "./types";

export type CitySceneCallbacks = {
  readonly onSelectLandmark: (id: LandmarkId | null) => void;
};

export type CitySceneController = {
  readonly setLighting: (mode: LightingMode) => void;
  readonly setPaused: (paused: boolean) => void;
  readonly setShowPins: (visible: boolean) => void;
  readonly setCrispOutlines: (visible: boolean) => void;
  readonly focusPreset: (preset: CameraPreset) => void;
  readonly focusLandmark: (id: LandmarkId) => void;
  readonly dispose: () => void;
};

type CameraGoal = {
  readonly target: THREE.Vector3;
  readonly zoom: number;
  readonly position?: THREE.Vector3;
};

const ORBIT_RADIUS = 42;
const FRUSTUM_HEIGHT = 25.8;

function sphericalPosition(
  target: THREE.Vector3,
  azimuth: number,
  polar: number,
): THREE.Vector3 {
  const sinP = Math.sin(polar);
  return new THREE.Vector3(
    target.x + ORBIT_RADIUS * sinP * Math.sin(azimuth),
    target.y + ORBIT_RADIUS * Math.cos(polar),
    target.z + ORBIT_RADIUS * sinP * Math.cos(azimuth),
  );
}

function isLandmarkId(value: unknown): value is LandmarkId {
  return (
    value === "golden-gate" ||
    value === "painted-ladies" ||
    value === "palace-of-fine-arts" ||
    value === "ferry-building" ||
    value === "coit-tower" ||
    value === "transamerica-pyramid" ||
    value === "salesforce-tower" ||
    value === "cable-car"
  );
}

export function createCityScene(
  container: HTMLElement,
  initialLighting: LightingMode,
  callbacks: CitySceneCallbacks,
): CitySceneController {
  const scene = new THREE.Scene();

  const width = Math.max(1, container.clientWidth);
  const height = Math.max(1, container.clientHeight);
  const aspect = width / height;

  const camera = new THREE.OrthographicCamera(
    (-FRUSTUM_HEIGHT * aspect) / 2,
    (FRUSTUM_HEIGHT * aspect) / 2,
    FRUSTUM_HEIGHT / 2,
    -FRUSTUM_HEIGHT / 2,
    0.1,
    180,
  );

  const overview = CAMERA_PRESETS[0];
  const initialTarget = new THREE.Vector3(
    overview ? overview.target[0] : 0,
    overview ? overview.target[1] : 1.3,
    overview ? overview.target[2] : 0,
  );
  const initialPos = sphericalPosition(
    initialTarget,
    overview ? overview.azimuth : Math.PI / 4,
    overview ? overview.polar : 0.96,
  );
  camera.position.copy(initialPos);
  camera.zoom = overview ? overview.zoom : 1.0;
  camera.lookAt(initialTarget);
  camera.updateProjectionMatrix();

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(width, height);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(initialTarget);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minZoom = 0.72;
  controls.maxZoom = 3.2;
  controls.minPolarAngle = 0.42;
  controls.maxPolarAngle = 1.24;
  controls.update();

  const ambientLight = new THREE.AmbientLight(0xfff8ee, 1.55);
  scene.add(ambientLight);

  const sunLight = new THREE.DirectionalLight(0xfff3de, 1.45);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.set(2048, 2048);
  sunLight.shadow.camera.left = -20;
  sunLight.shadow.camera.right = 20;
  sunLight.shadow.camera.top = 20;
  sunLight.shadow.camera.bottom = -20;
  sunLight.shadow.camera.near = 2;
  sunLight.shadow.camera.far = 90;
  sunLight.shadow.bias = -0.0005;
  scene.add(sunLight);

  const fillLight = new THREE.DirectionalLight(0xc9e0f5, 0.6);
  fillLight.position.set(-24, 18, -20);
  scene.add(fillLight);

  const terrain = buildTerrain();
  const landmarks = buildLandmarks();
  const blocks = buildCityBlocks();
  const foliage = buildFoliageAndProps();
  const traffic = buildTraffic();

  scene.add(terrain.group);
  scene.add(landmarks.group);
  scene.add(blocks);
  scene.add(foliage.group);
  scene.add(traffic.group);

  let paused = false;
  let simTime = 0;
  let lastFrame = performance.now();
  let cameraGoal: CameraGoal | null = null;
  let rafId = 0;

  function applyLighting(mode: LightingMode): void {
    const preset = LIGHTING_PRESETS[mode];
    scene.background = new THREE.Color(preset.background);
    ambientLight.color.setHex(preset.ambientColor);
    ambientLight.intensity = preset.ambientIntensity;
    sunLight.color.setHex(preset.sunColor);
    sunLight.intensity = preset.sunIntensity;
    sunLight.position.set(
      preset.sunPosition[0],
      preset.sunPosition[1],
      preset.sunPosition[2],
    );
    fillLight.color.setHex(preset.fillColor);
    fillLight.intensity = preset.fillIntensity;

    const lampColor = new THREE.Color(0xfff1c4).lerp(
      new THREE.Color(0xffb84d),
      preset.lampGlow,
    );
    for (const bulb of foliage.lampBulbs) {
      bulb.color.copy(lampColor);
    }

    const emissiveHex = mode === "night" ? 0x3b3522 : mode === "golden" ? 0x22170c : 0x000000;
    for (const mat of landmarks.windowMaterials) {
      mat.emissive.setHex(emissiveHex);
    }
  }

  applyLighting(initialLighting);
  logger.info("city scene initialized", { lighting: initialLighting });

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let pointerDownX = 0;
  let pointerDownY = 0;

  function onPointerDown(event: PointerEvent): void {
    pointerDownX = event.clientX;
    pointerDownY = event.clientY;
    cameraGoal = null;
  }

  function onPointerUp(event: PointerEvent): void {
    const moved = Math.hypot(
      event.clientX - pointerDownX,
      event.clientY - pointerDownY,
    );
    if (moved > 6) {
      return;
    }

    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(
      [foliage.pinsGroup, landmarks.group, blocks, traffic.group],
      true,
    );

    for (const hit of hits) {
      const maybeId: unknown = hit.object.userData.landmarkId;
      if (isLandmarkId(maybeId)) {
        callbacks.onSelectLandmark(maybeId);
        return;
      }
    }
  }

  function onResize(): void {
    const w = Math.max(1, container.clientWidth);
    const h = Math.max(1, container.clientHeight);
    const nextAspect = w / h;
    camera.left = (-FRUSTUM_HEIGHT * nextAspect) / 2;
    camera.right = (FRUSTUM_HEIGHT * nextAspect) / 2;
    camera.top = FRUSTUM_HEIGHT / 2;
    camera.bottom = -FRUSTUM_HEIGHT / 2;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  renderer.domElement.addEventListener("pointerup", onPointerUp);
  window.addEventListener("resize", onResize);

  function animate(now: number): void {
    rafId = window.requestAnimationFrame(animate);
    const dt = Math.min(0.05, (now - lastFrame) / 1000);
    lastFrame = now;

    if (!paused) {
      simTime += dt;
      traffic.update(simTime);
      terrain.waterTexture.offset.x = (simTime * 0.05) % 1;
      terrain.waterTexture.offset.y = (simTime * 0.025) % 1;

      for (const pin of foliage.pinSprites) {
        pin.sprite.position.y =
          pin.baseY + Math.sin(simTime * 2.4 + pin.phase) * 0.14;
      }
    }

    if (cameraGoal) {
      controls.target.lerp(cameraGoal.target, 0.09);
      camera.zoom = THREE.MathUtils.lerp(camera.zoom, cameraGoal.zoom, 0.09);
      if (cameraGoal.position) {
        camera.position.lerp(cameraGoal.position, 0.09);
      }
      camera.updateProjectionMatrix();

      const targetDist = controls.target.distanceTo(cameraGoal.target);
      const zoomDist = Math.abs(camera.zoom - cameraGoal.zoom);
      if (targetDist < 0.04 && zoomDist < 0.015) {
        cameraGoal = null;
      }
    }

    controls.update();
    renderer.render(scene, camera);
  }

  rafId = window.requestAnimationFrame(animate);

  return {
    setLighting(mode: LightingMode) {
      applyLighting(mode);
      logger.info("lighting changed", { mode });
    },
    setPaused(nextPaused: boolean) {
      paused = nextPaused;
    },
    setShowPins(visible: boolean) {
      foliage.pinsGroup.visible = visible;
    },
    setCrispOutlines(visible: boolean) {
      setOutlineVisibility(visible);
    },
    focusPreset(preset: CameraPreset) {
      const target = new THREE.Vector3(
        preset.target[0],
        preset.target[1],
        preset.target[2],
      );
      cameraGoal = {
        target,
        zoom: preset.zoom,
        position: sphericalPosition(target, preset.azimuth, preset.polar),
      };
    },
    focusLandmark(id: LandmarkId) {
      const landmark = findLandmarkById(id);
      const target = new THREE.Vector3(
        landmark.cameraTarget[0],
        landmark.cameraTarget[1],
        landmark.cameraTarget[2],
      );
      cameraGoal = {
        target,
        zoom: landmark.zoom,
        position: sphericalPosition(target, Math.PI / 4, 0.94),
      };
    },
    dispose() {
      window.cancelAnimationFrame(rafId);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("resize", onResize);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    },
  };
}
