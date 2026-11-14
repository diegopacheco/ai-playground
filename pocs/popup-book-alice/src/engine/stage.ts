import {
  ACESFilmicToneMapping,
  type Material,
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  Color,
  DirectionalLight,
  FogExp2,
  HemisphereLight,
  PCFSoftShadowMap,
  Plane,
  PerspectiveCamera,
  PointLight,
  Points,
  PointsMaterial,
  SRGBColorSpace,
  Scene,
  TextureLoader,
  Timer,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { seeded } from "../art/canvas";
import { SCENES } from "../scenes";
import type { Assets } from "../scenes/kit";
import { PAGES } from "../story/pages";
import { fitDistance, sceneZ, travelFrame } from "./timeline";
import type { Tunnel } from "./tunnel";

const FOV = 40;
const CLIP_X = 5.26;
const CLIP_Y = 3.2;
const TOP_SLOPE = 0.3;

function boxPlanes(frontZ: number): Plane[] {
  const top = new Plane(new Vector3(0, -1, -TOP_SLOPE), CLIP_Y + TOP_SLOPE * frontZ).normalize();
  return [new Plane(new Vector3(-1, 0, 0), CLIP_X), new Plane(new Vector3(1, 0, 0), CLIP_X), new Plane(new Vector3(0, 1, 0), CLIP_Y), top];
}
const MOTES = 700;

interface Travel {
  from: number;
  to: number;
  elapsed: number;
}

async function loadAssets(): Promise<Assets> {
  const loader = new TextureLoader();
  const [spark, disc] = await Promise.all([loader.loadAsync("assets/spark1.png"), loader.loadAsync("assets/disc.png")]);
  return { spark, disc };
}

export class Stage {
  private readonly renderer: WebGLRenderer;
  private readonly composer: EffectComposer;
  private readonly scene = new Scene();
  private readonly camera = new PerspectiveCamera(FOV, 1, 0.05, 200);
  private readonly clock = new Timer();
  private readonly cache = new Map<number, Tunnel>();
  private readonly pointer = new Vector2();
  private readonly pointerGoal = new Vector2();
  private readonly key = new DirectionalLight("#fff1d6", 2.6);
  private readonly lamp = new PointLight("#ffd9a0", 6, 22, 1.4);
  private assets?: Assets;
  private current = 0;
  private target = 0;
  private travel?: Travel;
  private frame = 0;
  private fit = 9.2;
  private disposed = false;

  constructor(private readonly host: HTMLElement, private readonly onReady: () => void) {
    this.renderer = new WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    this.renderer.localClippingEnabled = true;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFSoftShadowMap;
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.outputColorSpace = SRGBColorSpace;
    host.appendChild(this.renderer.domElement);
    this.scene.background = new Color("#140c1f");
    this.scene.fog = new FogExp2("#140c1f", 0.028);
    this.key.castShadow = true;
    this.key.shadow.mapSize.set(2048, 2048);
    this.key.shadow.camera.left = -8;
    this.key.shadow.camera.right = 8;
    this.key.shadow.camera.top = 6;
    this.key.shadow.camera.bottom = -6;
    this.key.shadow.camera.near = 1;
    this.key.shadow.camera.far = 40;
    this.key.shadow.bias = -0.0006;
    this.key.shadow.normalBias = 0.02;
    this.key.shadow.radius = 4;
    this.scene.add(this.key, this.key.target, this.lamp, new HemisphereLight("#cfe0ff", "#3a2440", 0.9));
    this.scene.add(this.motes());
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new UnrealBloomPass(new Vector2(256, 256), 0.35, 0.5, 0.88));
    this.composer.addPass(new OutputPass());
    window.addEventListener("resize", this.resize);
    window.addEventListener("pointermove", this.move);
    window.addEventListener("pointerup", this.release);
    window.addEventListener("pointercancel", this.release);
    this.resize();
  }

  async start(page: number): Promise<void> {
    await Promise.all([document.fonts.load("700 60px 'IM Fell English SC'"), document.fonts.load("italic 40px 'IM Fell English'"), document.fonts.load("700 40px 'Fredericka the Great'")]);
    this.assets = await loadAssets();
    if (this.disposed) return;
    this.current = page - 1;
    this.target = this.current;
    this.travel = { from: this.current, to: this.current, elapsed: 0 };
    this.tunnelAt(this.current).setOpenness(0);
    this.loop();
    this.onReady();
  }

  goTo(page: number): void {
    this.target = page - 1;
  }

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.frame);
    window.removeEventListener("resize", this.resize);
    window.removeEventListener("pointermove", this.move);
    window.removeEventListener("pointerup", this.release);
    window.removeEventListener("pointercancel", this.release);
    for (const tunnel of this.cache.values()) tunnel.dispose();
    this.composer.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  private tunnelAt(index: number): Tunnel {
    const cached = this.cache.get(index);
    if (cached) return cached;
    const page = PAGES[index];
    if (!page || !this.assets) throw new Error(`no page at ${index}`);
    const tunnel = SCENES[page.scene](sceneZ(index), this.assets);
    const planes = page.scene === "fall" ? [] : boxPlanes(sceneZ(index));
    tunnel.group.traverse((child) => {
      const material = (child as { material?: Material | Material[] }).material;
      for (const m of Array.isArray(material) ? material : material ? [material] : []) m.clippingPlanes = planes;
    });
    this.scene.add(tunnel.group);
    this.cache.set(index, tunnel);
    return tunnel;
  }

  private loop = (): void => {
    if (this.disposed) return;
    this.frame = requestAnimationFrame(this.loop);
    this.clock.update();
    const dt = Math.min(this.clock.getDelta(), 0.05);
    if (!this.travel && this.target !== this.current) {
      this.travel = { from: this.current, to: this.target, elapsed: 0 };
      this.tunnelAt(this.target).setOpenness(0);
    }
    let cameraZ = sceneZ(this.current) + this.fit;
    if (this.travel) {
      const travel = this.travel;
      travel.elapsed += dt;
      const frame = travelFrame(travel.elapsed);
      cameraZ = sceneZ(travel.from) + this.fit + (sceneZ(travel.to) - sceneZ(travel.from)) * frame.progress;
      this.tunnelAt(travel.to).setOpenness(frame.nextOpenness);
      if (frame.arrived) {
        this.current = travel.to;
        this.travel = undefined;
      }
    }
    const visible = new Set([this.current, this.travel?.from ?? this.current, this.travel?.to ?? this.current]);
    for (const [index, tunnel] of this.cache) {
      tunnel.group.visible = visible.has(index);
      if (!tunnel.group.visible) continue;
      tunnel.update(dt);
      tunnel.fade(cameraZ);
    }
    this.pointer.lerp(this.pointerGoal, 0.06);
    const focusZ = cameraZ - this.fit;
    this.camera.position.set(this.pointer.x * 1.1, -this.pointer.y * 0.6, cameraZ);
    this.camera.lookAt(this.pointer.x * 0.25, -this.pointer.y * 0.15, focusZ - 3);
    this.key.position.set(-3.5, 4.5, focusZ + 16);
    this.key.target.position.set(0, -1, focusZ - 4);
    this.lamp.position.set(2, 1.5, focusZ + 8);
    this.composer.render(dt);
  };

  private motes(): Points {
    const rand = seeded(4);
    const positions = new Float32Array(MOTES * 3);
    for (let i = 0; i < MOTES; i++) {
      positions[i * 3] = (rand() - 0.5) * 14;
      positions[i * 3 + 1] = (rand() - 0.5) * 9;
      positions[i * 3 + 2] = 12 - rand() * (PAGES.length * 16 + 10);
    }
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(positions, 3));
    const points = new Points(geometry, new PointsMaterial({ color: "#ffe6b0", size: 0.05, transparent: true, opacity: 0.8, blending: AdditiveBlending, depthWrite: false }));
    return points;
  }

  private resize = (): void => {
    const { clientWidth, clientHeight } = this.host;
    this.camera.aspect = clientWidth / Math.max(clientHeight, 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(clientWidth, clientHeight);
    this.composer.setSize(clientWidth, clientHeight);
    this.fit = fitDistance(this.camera.aspect, this.camera.aspect < 1 ? 3.0 : 5.7, this.camera.aspect < 1 ? 4.6 : 3.9, FOV);
  };

  private move = (event: PointerEvent): void => {
    this.pointerGoal.set((event.clientX / window.innerWidth) * 2 - 1, (event.clientY / window.innerHeight) * 2 - 1);
  };

  private release = (event: PointerEvent): void => {
    if (event.pointerType !== "mouse") this.pointerGoal.set(0, 0);
  };
}

