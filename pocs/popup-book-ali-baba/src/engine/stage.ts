import {
  ACESFilmicToneMapping,
  Color,
  FogExp2,
  LinearFilter,
  PerspectiveCamera,
  SRGBColorSpace,
  Scene,
  TextureLoader,
  Timer,
  Vector2,
  Vector3,
  WebGLRenderer,
  WebGLRenderTarget,
} from "three";
import { RectAreaLightUniformsLib } from "three/examples/jsm/lights/RectAreaLightUniformsLib.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { SCENES } from "../scenes";
import type { Assets } from "../scenes/kit";
import { PAGES } from "../story/pages";
import type { Act } from "./act";
import { MASK_LAYER, lampUniform } from "./materials";
import { Room, lampTarget } from "./room";
import { SCREEN_H, SCREEN_W, lampFrustum } from "./shadow";
import { changeFrame, clamp01, easeInOutCubic } from "./timeline";

const MASK_W = 1280;
const NEAR = 0.05;
const FRONT_LOOK = new Vector3(0, 2.35, 0);
const PEEK_EYE = new Vector3(7.4, 3.9, -7.6);
const PEEK_LOOK = new Vector3(0, 1.7, -1.4);

interface Change {
  from: number;
  to: number;
  elapsed: number;
  swapped: boolean;
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
  private readonly camera = new PerspectiveCamera(36, 1, 0.1, 120);
  private readonly maskCamera = new PerspectiveCamera();
  private readonly mask = new WebGLRenderTarget(MASK_W, Math.round((MASK_W * SCREEN_H) / SCREEN_W), { minFilter: LinearFilter, magFilter: LinearFilter });
  private readonly clock = new Timer();
  private readonly cache = new Map<number, Act>();
  private readonly pointer = new Vector3();
  private readonly pointerGoal = new Vector3();
  private readonly motion = new Vector3();
  private readonly white = new Color("#ffffff");
  private room?: Room;
  private assets?: Assets;
  private current = 0;
  private target = 0;
  private change?: Change;
  private rising = 1;
  private level = 0;
  private peek = 0;
  private peekGoal = 0;
  private frame = 0;
  private disposed = false;

  constructor(private readonly host: HTMLElement, private readonly onReady: () => void) {
    this.renderer = new WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.outputColorSpace = SRGBColorSpace;
    host.appendChild(this.renderer.domElement);
    RectAreaLightUniformsLib.init();
    this.scene.background = new Color("#0b0610");
    this.scene.fog = new FogExp2("#0b0610", 0.022);
    this.maskCamera.layers.set(MASK_LAYER);
    this.maskCamera.rotation.set(0, Math.PI, 0);
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new UnrealBloomPass(new Vector2(256, 256), 0.32, 0.45, 0.93));
    this.composer.addPass(new OutputPass());
    window.addEventListener("resize", this.resize);
    window.addEventListener("pointermove", this.move);
    window.addEventListener("pointerup", this.release);
    window.addEventListener("pointercancel", this.release);
    this.resize();
  }

  async start(page: number): Promise<void> {
    await Promise.all([
      document.fonts.load("900 60px 'Cinzel Decorative'"),
      document.fonts.load("700 40px 'Cinzel Decorative'"),
      document.fonts.load("700 40px Amiri"),
    ]);
    this.assets = await loadAssets();
    if (this.disposed) return;
    this.room = new Room(this.mask.texture, this.assets.spark);
    this.scene.add(this.room.group);
    this.current = page - 1;
    this.target = this.current;
    const first = this.actAt(this.current);
    first.setOpenness(0);
    this.room.setTint(first.top, first.bottom);
    this.rising = 0;
    this.loop();
    this.onReady();
  }

  goTo(page: number): void {
    this.target = page - 1;
  }

  setPeek(on: boolean): void {
    this.peekGoal = on ? 1 : 0;
  }

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.frame);
    window.removeEventListener("resize", this.resize);
    window.removeEventListener("pointermove", this.move);
    window.removeEventListener("pointerup", this.release);
    window.removeEventListener("pointercancel", this.release);
    for (const act of this.cache.values()) act.dispose();
    this.mask.dispose();
    this.composer.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  private actAt(index: number): Act {
    const cached = this.cache.get(index);
    if (cached) return cached;
    const page = PAGES[index];
    if (!page || !this.assets) throw new Error(`no page at ${index}`);
    const act = SCENES[page.scene](this.assets);
    act.group.visible = false;
    this.scene.add(act.group);
    this.cache.set(index, act);
    return act;
  }

  private stepChange(dt: number): void {
    const change = this.change;
    if (!change) {
      if (this.rising < 1) {
        this.rising = Math.min(this.rising + dt / 1.8, 1);
        this.actAt(this.current).setOpenness(this.rising);
        this.level = Math.min(1, this.rising * 1.6);
      } else this.level = 1;
      return;
    }
    change.elapsed += dt;
    const frame = changeFrame(change.elapsed);
    this.level = frame.level;
    if (!change.swapped) {
      this.actAt(change.from).setOpenness(frame.oldOpenness);
      if (frame.swapped) {
        change.swapped = true;
        this.actAt(change.from).setOpenness(0);
        change.to = this.target;
        this.current = this.target;
        const next = this.actAt(this.current);
        this.room?.setTint(next.top, next.bottom);
      }
    } else {
      this.actAt(change.to).setOpenness(frame.newOpenness);
    }
    if (frame.done) {
      this.change = undefined;
      this.rising = 1;
    }
  }

  private loop = (): void => {
    if (this.disposed || !this.room) return;
    this.frame = requestAnimationFrame(this.loop);
    this.clock.update();
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const time = this.clock.getElapsed();
    if (!this.change && this.target !== this.current) this.change = { from: this.current, to: this.current, elapsed: 0, swapped: false };
    this.stepChange(dt);
    const active = this.actAt(this.current);
    for (const [index, act] of this.cache) {
      if (index === this.current || index === this.change?.from) act.update(dt);
    }
    this.motion.set(0, 0, 0);
    active.lampMotion(active.clock, this.motion);
    this.pointer.lerp(this.pointerGoal, 0.06);
    lampUniform.value.lerp(lampTarget(active.lamp, this.motion, this.pointer), 0.12);
    const level = this.level * active.levelAt(active.clock);
    this.room.setTint(active.top, active.bottom);
    this.room.update(time, level);
    this.renderMask();
    this.aim(dt, time);
    this.composer.render(dt);
  };

  private renderMask(): void {
    const lamp = lampUniform.value;
    const f = lampFrustum(lamp, NEAR);
    this.maskCamera.position.copy(lamp);
    this.maskCamera.projectionMatrix.makePerspective(f.left, f.right, f.top, f.bottom, NEAR, 40);
    this.maskCamera.projectionMatrixInverse.copy(this.maskCamera.projectionMatrix).invert();
    const background = this.scene.background;
    const fog = this.scene.fog;
    this.scene.background = this.white;
    this.scene.fog = null;
    this.renderer.setRenderTarget(this.mask);
    this.renderer.setClearColor(this.white, 1);
    this.renderer.clear();
    this.renderer.render(this.scene, this.maskCamera);
    this.renderer.setRenderTarget(null);
    this.scene.background = background;
    this.scene.fog = fog;
  }

  private aim(dt: number, time: number): void {
    this.peek += (this.peekGoal - this.peek) * Math.min(dt * 2.2, 1);
    const aspect = this.camera.aspect;
    const half = Math.tan((this.camera.fov * Math.PI) / 360);
    const halfW = aspect < 1 ? 4.3 : 5.4;
    const halfH = aspect < 1 ? 3.2 : 4.2;
    const distance = Math.max(halfW / (half * aspect), halfH / half);
    const sway = Math.sin(time * 0.13) * 0.25;
    const front = new Vector3(sway, FRONT_LOOK.y + distance * 0.04, distance);
    const look = FRONT_LOOK.clone();
    if (aspect < 1) look.y = 2.2;
    const t = easeInOutCubic(clamp01(this.peek));
    const arc = Math.sin(t * Math.PI) * 4;
    this.camera.position.copy(front).lerp(PEEK_EYE, t);
    this.camera.position.x += arc;
    this.camera.lookAt(look.lerp(PEEK_LOOK, t));
  }

  private resize = (): void => {
    const { clientWidth, clientHeight } = this.host;
    this.camera.aspect = clientWidth / Math.max(clientHeight, 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(clientWidth, clientHeight);
    this.composer.setSize(clientWidth, clientHeight);
  };

  private move = (event: PointerEvent): void => {
    const x = (event.clientX / window.innerWidth) * 2 - 1;
    const y = (event.clientY / window.innerHeight) * 2 - 1;
    this.pointerGoal.set(x * 1.4, -y * 0.5, y > 0 ? y * 1.5 : y * 1.1);
  };

  private release = (event: PointerEvent): void => {
    if (event.pointerType !== "mouse") this.pointerGoal.set(0, 0, 0);
  };
}
