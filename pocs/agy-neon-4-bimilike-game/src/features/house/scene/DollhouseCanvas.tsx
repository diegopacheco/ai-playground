import { useEffect, useRef } from "react";
import * as THREE from "three";
import { sound } from "../../../lib/sound";
import { getCharacterSpec } from "../../characters/catalog";
import { CHARACTER_IDS, type CharacterId } from "../../characters/types";
import { clampHousePosition } from "../catalog";
import {
  SURFACE_IDS,
  type HouseState,
  type RoomFocus,
  type SelectionTarget,
  type SurfaceId,
  type ViewMode,
} from "../types";
import { buildCharacterMesh } from "./buildCharacter";
import { buildDollhouseShell, type DollhouseSurfaceRefs } from "./buildDollhouse";
import { buildFoodMesh } from "./buildFood";
import { buildFurnitureMesh } from "./buildFurniture";

type HitTarget =
  | { readonly kind: "furniture"; readonly id: string }
  | { readonly kind: "character"; readonly id: CharacterId }
  | { readonly kind: "surface"; readonly id: SurfaceId };

type BurstParticle = {
  readonly mesh: THREE.Mesh;
  vx: number;
  vy: number;
  vz: number;
  life: number;
};

type DollhouseCanvasProps = {
  readonly state: HouseState;
  readonly roomFocus: RoomFocus;
  readonly viewMode: ViewMode;
  readonly selection: SelectionTarget | null;
  readonly feedingDishUid: string | null;
  readonly burstTick: number;
  readonly onSelect: (target: SelectionTarget | null) => void;
  readonly onMoveFurniture: (id: string, x: number, z: number) => void;
  readonly onMoveCharacter: (id: CharacterId, x: number, z: number) => void;
  readonly onTapFurnitureSpecial: (id: string) => void;
  readonly onTapCharacter: (id: CharacterId) => void;
};

function readHitTarget(object: THREE.Object3D): HitTarget | null {
  let curr: THREE.Object3D | null = object;
  while (curr) {
    const data: Record<string, unknown> = curr.userData;
    const hitKind = data["hitKind"];
    const hitId = data["hitId"];
    if (hitKind === "furniture" && typeof hitId === "string") {
      return { kind: "furniture", id: hitId };
    }
    if (hitKind === "character" && typeof hitId === "string") {
      const matched = CHARACTER_IDS.find((id) => id === hitId);
      if (matched) return { kind: "character", id: matched };
    }
    if (hitKind === "surface" && typeof hitId === "string") {
      const matched = SURFACE_IDS.find((id) => id === hitId);
      if (matched) return { kind: "surface", id: matched };
    }
    curr = curr.parent;
  }
  return null;
}

function focusTargetFor(room: RoomFocus): { readonly x: number; readonly y: number; readonly dist: number; readonly orthoHalf: number } {
  switch (room) {
    case "kitchen":
      return { x: -4.4, y: 1.5, dist: 7.4, orthoHalf: 3.1 };
    case "living":
      return { x: 0.0, y: 1.5, dist: 7.4, orthoHalf: 3.1 };
    case "bedroom":
      return { x: 4.4, y: 1.5, dist: 7.4, orthoHalf: 3.1 };
    case "all":
      return { x: 0.0, y: 2.0, dist: 15.6, orthoHalf: 5.2 };
    default: {
      const unreachable: never = room;
      return unreachable;
    }
  }
}

export function DollhouseCanvas({
  state,
  roomFocus,
  viewMode,
  selection,
  feedingDishUid,
  burstTick,
  onSelect,
  onMoveFurniture,
  onMoveCharacter,
  onTapFurnitureSpecial,
  onTapCharacter,
}: DollhouseCanvasProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const stateRef = useRef(state);
  const roomRef = useRef(roomFocus);
  const viewModeRef = useRef(viewMode);
  const selectionRef = useRef(selection);
  const feedingRef = useRef(feedingDishUid);
  const callbacksRef = useRef({
    onSelect,
    onMoveFurniture,
    onMoveCharacter,
    onTapFurnitureSpecial,
    onTapCharacter,
  });

  stateRef.current = state;
  roomRef.current = roomFocus;
  viewModeRef.current = viewMode;
  selectionRef.current = selection;
  feedingRef.current = feedingDishUid;
  callbacksRef.current = {
    onSelect,
    onMoveFurniture,
    onMoveCharacter,
    onTapFurnitureSpecial,
    onTapCharacter,
  };

  const sceneBundleRef = useRef<{
    scene: THREE.Scene;
    shell: DollhouseSurfaceRefs;
    furnitureRoot: THREE.Group;
    charactersRoot: THREE.Group;
    foodRoot: THREE.Group;
    particlesRoot: THREE.Group;
    selectionRing: THREE.Mesh;
    furnitureMap: Map<string, THREE.Group>;
    characterMap: Map<CharacterId, THREE.Group>;
    particles: BurstParticle[];
    spawnBurst: (x: number, y: number, z: number, color: string) => void;
  } | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#bfe9ff");

    const perspCam = new THREE.PerspectiveCamera(40, 1, 0.1, 80);
    perspCam.position.set(0, 5.4, 15.6);

    const orthoCam = new THREE.OrthographicCamera(-8, 8, 5, -5, 0.1, 80);
    orthoCam.position.set(0, 2.2, 18);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    host.appendChild(renderer.domElement);

    const hemi = new THREE.HemisphereLight("#fffdf5", "#b5e48c", 1.15);
    scene.add(hemi);

    const sun = new THREE.DirectionalLight("#fff5d6", 1.45);
    sun.position.set(7.5, 13.5, 11);
    sun.castShadow = true;
    sun.shadow.mapSize.width = 1024;
    sun.shadow.mapSize.height = 1024;
    sun.shadow.camera.near = 2;
    sun.shadow.camera.far = 36;
    sun.shadow.camera.left = -10;
    sun.shadow.camera.right = 10;
    sun.shadow.camera.top = 9;
    sun.shadow.camera.bottom = -6;
    sun.shadow.bias = -0.0005;
    scene.add(sun);

    const fill = new THREE.DirectionalLight("#ffd6e8", 0.45);
    fill.position.set(-8, 5, 7);
    scene.add(fill);

    const shell = buildDollhouseShell(stateRef.current.surfaces);
    scene.add(shell.group);

    const furnitureRoot = new THREE.Group();
    const charactersRoot = new THREE.Group();
    const foodRoot = new THREE.Group();
    const particlesRoot = new THREE.Group();
    scene.add(furnitureRoot, charactersRoot, foodRoot, particlesRoot);

    const selectionRing = new THREE.Mesh(
      new THREE.RingGeometry(0.62, 0.78, 28),
      new THREE.MeshBasicMaterial({ color: "#ff5d8f", side: THREE.DoubleSide }),
    );
    selectionRing.rotation.x = -Math.PI / 2;
    selectionRing.position.y = 0.04;
    selectionRing.visible = false;
    scene.add(selectionRing);

    const furnitureMap = new Map<string, THREE.Group>();
    const characterMap = new Map<CharacterId, THREE.Group>();
    const particles: BurstParticle[] = [];

    const particleGeo = new THREE.OctahedronGeometry(0.11, 0);
    const spawnBurst = (x: number, y: number, z: number, color: string) => {
      const palette = [color, "#ffe066", "#ff8fab", "#72efdd"];
      for (let i = 0; i < 10; i += 1) {
        const mat = new THREE.MeshBasicMaterial({ color: palette[i % palette.length] ?? "#ffe066" });
        const mesh = new THREE.Mesh(particleGeo, mat);
        mesh.position.set(x, y, z);
        particlesRoot.add(mesh);
        const angle = (i / 10) * Math.PI * 2;
        const speed = 0.045 + (i % 3) * 0.018;
        particles.push({
          mesh,
          vx: Math.cos(angle) * speed,
          vy: 0.065 + (i % 2) * 0.03,
          vz: Math.sin(angle) * speed,
          life: 1.0,
        });
      }
    };

    sceneBundleRef.current = {
      scene,
      shell,
      furnitureRoot,
      charactersRoot,
      foodRoot,
      particlesRoot,
      selectionRing,
      furnitureMap,
      characterMap,
      particles,
      spawnBurst,
    };

    const raycaster = new THREE.Raycaster();
    const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const pointerNdc = new THREE.Vector2();
    const hitPoint = new THREE.Vector3();

    let dragState:
      | {
          kind: "furniture";
          id: string;
          startX: number;
          startY: number;
          moved: boolean;
          offsetX: number;
          offsetZ: number;
        }
      | {
          kind: "character";
          id: CharacterId;
          startX: number;
          startY: number;
          moved: boolean;
          offsetX: number;
          offsetZ: number;
        }
      | null = null;

    const getActiveCamera = (): THREE.Camera =>
      viewModeRef.current === "2d" ? orthoCam : perspCam;

    const updateNdc = (clientX: number, clientY: number) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointerNdc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
      pointerNdc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    };

    const onPointerDown = (ev: PointerEvent) => {
      updateNdc(ev.clientX, ev.clientY);
      raycaster.setFromCamera(pointerNdc, getActiveCamera());

      const charHits = raycaster.intersectObjects(charactersRoot.children, true);
      if (charHits[0]) {
        const target = readHitTarget(charHits[0].object);
        if (target && target.kind === "character") {
          if (feedingRef.current) {
            const mesh = characterMap.get(target.id);
            if (mesh) spawnBurst(mesh.position.x, 1.1, mesh.position.z, "#ff6b8b");
            callbacksRef.current.onTapCharacter(target.id);
            return;
          }
          const mesh = characterMap.get(target.id);
          if (mesh && raycaster.ray.intersectPlane(floorPlane, hitPoint)) {
            renderer.domElement.setPointerCapture(ev.pointerId);
            dragState = {
              kind: "character",
              id: target.id,
              startX: ev.clientX,
              startY: ev.clientY,
              moved: false,
              offsetX: mesh.position.x - hitPoint.x,
              offsetZ: mesh.position.z - hitPoint.z,
            };
          }
          return;
        }
      }

      const furnHits = raycaster.intersectObjects(furnitureRoot.children, true);
      if (furnHits[0]) {
        const target = readHitTarget(furnHits[0].object);
        if (target && target.kind === "furniture") {
          const mesh = furnitureMap.get(target.id);
          if (mesh && raycaster.ray.intersectPlane(floorPlane, hitPoint)) {
            renderer.domElement.setPointerCapture(ev.pointerId);
            dragState = {
              kind: "furniture",
              id: target.id,
              startX: ev.clientX,
              startY: ev.clientY,
              moved: false,
              offsetX: mesh.position.x - hitPoint.x,
              offsetZ: mesh.position.z - hitPoint.z,
            };
          }
          return;
        }
      }

      const shellHits = raycaster.intersectObjects(shell.group.children, true);
      if (shellHits[0]) {
        const target = readHitTarget(shellHits[0].object);
        if (target && target.kind === "surface") {
          sound.pop();
          callbacksRef.current.onSelect({ kind: "surface", id: target.id });
          return;
        }
      }

      callbacksRef.current.onSelect(null);
    };

    const onPointerMove = (ev: PointerEvent) => {
      if (!dragState) return;
      const dx = ev.clientX - dragState.startX;
      const dy = ev.clientY - dragState.startY;
      if (!dragState.moved && Math.hypot(dx, dy) > 8) {
        dragState.moved = true;
        sound.pickup();
      }
      if (!dragState.moved) return;

      updateNdc(ev.clientX, ev.clientY);
      raycaster.setFromCamera(pointerNdc, getActiveCamera());
      if (raycaster.ray.intersectPlane(floorPlane, hitPoint)) {
        const clamped = clampHousePosition(hitPoint.x + dragState.offsetX, hitPoint.z + dragState.offsetZ);
        if (dragState.kind === "furniture") {
          const mesh = furnitureMap.get(dragState.id);
          if (mesh) {
            mesh.position.set(clamped.x, 0.22, clamped.z);
          }
        } else {
          const mesh = characterMap.get(dragState.id);
          if (mesh) {
            mesh.position.set(clamped.x, 0.25, clamped.z);
          }
        }
      }
    };

    const onPointerUp = (ev: PointerEvent) => {
      if (!dragState) return;
      const activeDrag = dragState;
      dragState = null;
      if (renderer.domElement.hasPointerCapture(ev.pointerId)) {
        renderer.domElement.releasePointerCapture(ev.pointerId);
      }

      if (activeDrag.moved) {
        sound.drop();
        if (activeDrag.kind === "furniture") {
          const mesh = furnitureMap.get(activeDrag.id);
          if (mesh) {
            mesh.position.y = 0;
            spawnBurst(mesh.position.x, 0.7, mesh.position.z, "#72efdd");
            callbacksRef.current.onMoveFurniture(activeDrag.id, mesh.position.x, mesh.position.z);
          }
        } else {
          const mesh = characterMap.get(activeDrag.id);
          if (mesh) {
            mesh.position.y = 0;
            spawnBurst(mesh.position.x, 0.8, mesh.position.z, "#ffe066");
            callbacksRef.current.onMoveCharacter(activeDrag.id, mesh.position.x, mesh.position.z);
          }
        }
      } else {
        if (activeDrag.kind === "furniture") {
          callbacksRef.current.onSelect({ kind: "furniture", id: activeDrag.id });
          callbacksRef.current.onTapFurnitureSpecial(activeDrag.id);
        } else {
          const mesh = characterMap.get(activeDrag.id);
          if (mesh) spawnBurst(mesh.position.x, 1.0, mesh.position.z, "#ff8fab");
          callbacksRef.current.onSelect({ kind: "character", id: activeDrag.id });
          callbacksRef.current.onTapCharacter(activeDrag.id);
        }
      }
    };

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("pointercancel", onPointerUp);

    let width = host.clientWidth || 800;
    let height = host.clientHeight || 500;
    const syncSize = () => {
      width = Math.max(1, host.clientWidth);
      height = Math.max(1, host.clientHeight);
      renderer.setSize(width, height, false);
      perspCam.aspect = width / height;
      perspCam.updateProjectionMatrix();
    };
    syncSize();
    const resizeObserver = new ResizeObserver(syncSize);
    resizeObserver.observe(host);

    const lookCurrent = new THREE.Vector3(0, 2.0, -0.3);
    const lookTarget = new THREE.Vector3(0, 2.0, -0.3);
    const clock = new THREE.Clock();
    let frameId = 0;

    const animate = () => {
      frameId = window.requestAnimationFrame(animate);
      const t = clock.getElapsedTime();
      const aspect = width / height;
      const focus = focusTargetFor(roomRef.current);

      const mobileZoomOut = aspect < 1.05 ? 1.55 / Math.max(aspect, 0.48) : aspect < 1.45 ? 1.18 : 1.0;
      const desiredPerspX = focus.x;
      const desiredPerspY = roomRef.current === "all" ? 4.3 : 3.1;
      const desiredPerspZ = focus.dist * mobileZoomOut;

      perspCam.position.x += (desiredPerspX - perspCam.position.x) * 0.08;
      perspCam.position.y += (desiredPerspY - perspCam.position.y) * 0.08;
      perspCam.position.z += (desiredPerspZ - perspCam.position.z) * 0.08;
      lookTarget.set(focus.x, focus.y, -0.3);
      lookCurrent.lerp(lookTarget, 0.1);
      perspCam.lookAt(lookCurrent);

      const halfH = focus.orthoHalf * (aspect < 1.25 ? 1.28 / Math.max(aspect, 0.52) : 1.0);
      const halfW = halfH * aspect;
      orthoCam.left = -halfW;
      orthoCam.right = halfW;
      orthoCam.top = halfH;
      orthoCam.bottom = -halfH;
      orthoCam.position.x += (focus.x - orthoCam.position.x) * 0.1;
      orthoCam.position.y += ((focus.y + 0.4) - orthoCam.position.y) * 0.1;
      orthoCam.lookAt(orthoCam.position.x, orthoCam.position.y - 0.15, 0);
      orthoCam.updateProjectionMatrix();

      shell.clouds.forEach((cloud, idx) => {
        cloud.position.x = ((cloud.position.x + 0.006 * (idx + 1) + 12) % 24) - 12;
      });

      const now = Date.now();
      for (const placement of stateRef.current.characters) {
        const mesh = characterMap.get(placement.id);
        if (!mesh) continue;
        const actor = mesh.getObjectByName("actor");
        const head = mesh.getObjectByName("head");
        const rightArm = mesh.getObjectByName("rightArm");
        const phase = placement.id.charCodeAt(0) * 0.7;
        const activeMood = placement.moodUntil > now ? placement.mood : "happy";

        if (actor) {
          if (activeMood === "dancing") {
            actor.position.y = Math.abs(Math.sin(t * 7 + phase)) * 0.28;
            actor.rotation.z = Math.sin(t * 9 + phase) * 0.16;
          } else if (activeMood === "yummy") {
            actor.position.y = Math.abs(Math.sin(t * 10 + phase)) * 0.18;
            actor.scale.set(
              1 + Math.sin(t * 12) * 0.06,
              1 - Math.sin(t * 12) * 0.05,
              1,
            );
            actor.rotation.z = 0;
          } else {
            actor.position.y = Math.sin(t * 2.6 + phase) * 0.04;
            actor.rotation.z = Math.sin(t * 1.8 + phase) * 0.03;
            actor.scale.set(1, 1, 1);
          }
        }
        if (head) {
          head.rotation.y = Math.sin(t * 1.4 + phase) * 0.18;
        }
        if (rightArm) {
          rightArm.rotation.z =
            activeMood === "waving" || activeMood === "dancing"
              ? -2.2 + Math.sin(t * 11 + phase) * 0.45
              : -0.45 + Math.sin(t * 2.2 + phase) * 0.1;
        }
      }

      const phoneGroup = furnitureMap.get("phoneTable");
      if (phoneGroup) {
        const handset = phoneGroup.getObjectByName("phoneHandset");
        if (handset) {
          handset.position.y = 0.98 + Math.abs(Math.sin(t * 3.5)) * 0.04;
          handset.rotation.y = Math.sin(t * 6) * 0.08;
        }
      }

      const sel = selectionRef.current;
      if (sel && sel.kind === "furniture") {
        const fm = furnitureMap.get(sel.id);
        if (fm) {
          selectionRing.visible = true;
          selectionRing.position.set(fm.position.x, 0.05, fm.position.z);
          const pulse = 1 + Math.sin(t * 6) * 0.08;
          selectionRing.scale.set(pulse, pulse, 1);
        } else {
          selectionRing.visible = false;
        }
      } else if (sel && sel.kind === "character") {
        const cm = characterMap.get(sel.id);
        if (cm) {
          selectionRing.visible = true;
          selectionRing.position.set(cm.position.x, 0.05, cm.position.z);
          const pulse = 0.78 + Math.sin(t * 6) * 0.07;
          selectionRing.scale.set(pulse, pulse, 1);
        } else {
          selectionRing.visible = false;
        }
      } else {
        selectionRing.visible = false;
      }

      for (let i = particles.length - 1; i >= 0; i -= 1) {
        const p = particles[i];
        if (!p) continue;
        p.mesh.position.x += p.vx;
        p.mesh.position.y += p.vy;
        p.mesh.position.z += p.vz;
        p.vy -= 0.0028;
        p.mesh.rotation.x += 0.12;
        p.mesh.rotation.y += 0.14;
        p.life -= 0.025;
        p.mesh.scale.setScalar(Math.max(0.01, p.life));
        if (p.life <= 0) {
          particlesRoot.remove(p.mesh);
          particles.splice(i, 1);
        }
      }

      renderer.render(scene, getActiveCamera());
    };
    animate();

    return () => {
      window.cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("pointercancel", onPointerUp);
      renderer.dispose();
      if (renderer.domElement.parentElement === host) {
        host.removeChild(renderer.domElement);
      }
      sceneBundleRef.current = null;
    };
  }, []);

  useEffect(() => {
    const bundle = sceneBundleRef.current;
    if (!bundle) return;

    for (const sid of SURFACE_IDS) {
      bundle.shell.materials[sid].color.set(state.surfaces[sid]);
    }

    bundle.furnitureRoot.clear();
    bundle.furnitureMap.clear();
    for (const item of state.furniture) {
      const mesh = buildFurnitureMesh(item);
      mesh.position.set(item.x, 0, item.z);
      bundle.furnitureRoot.add(mesh);
      bundle.furnitureMap.set(item.id, mesh);
    }

    bundle.charactersRoot.clear();
    bundle.characterMap.clear();
    for (const placement of state.characters) {
      const spec = getCharacterSpec(placement.id);
      const mesh = buildCharacterMesh(spec);
      mesh.position.set(placement.x, 0, placement.z);
      bundle.charactersRoot.add(mesh);
      bundle.characterMap.set(placement.id, mesh);
    }

    bundle.foodRoot.clear();
    const tableItem = state.furniture.find((f) => f.id === "table");
    const tx = tableItem?.x ?? -3.55;
    const tz = tableItem?.z ?? 0.45;
    const offsets: readonly (readonly [number, number])[] = [
      [0, 0],
      [-0.42, -0.2],
      [0.42, -0.2],
      [-0.36, 0.28],
      [0.36, 0.28],
    ];
    for (const dish of state.dishes) {
      const offset = offsets[dish.slotIndex % offsets.length] ?? [0, 0];
      const foodMesh = buildFoodMesh(dish);
      foodMesh.position.set(tx + offset[0], 0.86, tz + offset[1]);
      bundle.foodRoot.add(foodMesh);
    }
  }, [state]);

  useEffect(() => {
    if (burstTick === 0) return;
    const bundle = sceneBundleRef.current;
    if (!bundle) return;
    bundle.spawnBurst(-3.55, 1.3, 0.45, "#ffe066");
    bundle.spawnBurst(0, 1.5, 0.4, "#ff8fab");
  }, [burstTick]);

  return (
    <div
      ref={hostRef}
      className="dollhouse-canvas"
      aria-label="Interactive 3D and 2D Dollhouse"
    />
  );
}
