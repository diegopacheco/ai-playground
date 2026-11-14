import {
  ACESFilmicToneMapping,
  Color,
  FogExp2,
  PCFShadowMap,
  PerspectiveCamera,
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
import { SCENES } from "../scenes";
import type { Assets } from "../scenes/kit";
import { PAGES } from "../story/pages";
import type { PopupScene } from "./popup";
import { Theater } from "./theater";
import { changeFrame } from "./timeline";

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
  private readonly bloom: UnrealBloomPass;
  private readonly scene = new Scene();
  private readonly camera = new PerspectiveCamera(36, 1, 0.1, 100);
  private readonly theater = new Theater();
  private readonly clock = new Timer();
  private readonly cache = new Map<number, PopupScene>();
  private readonly pointer = new Vector2();
  private readonly look = new Vector3(0, 3.0, 0);
  private assets?: Assets;
  private current = 0;
  private target = 0;
  private change?: Change;
  private rising = 1;
  private frame = 0;
  private disposed = false;
  private snapped = false;

  constructor(private readonly host: HTMLElement, private readonly onReady: () => void) {
    this.renderer = new WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFShadowMap;
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.outputColorSpace = SRGBColorSpace;
    host.appendChild(this.renderer.domElement);
    this.scene.background = new Color("#12050a");
    this.scene.fog = new FogExp2("#12050a", 0.018);
    this.scene.add(this.theater.group);
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new Vector2(256, 256), 0.45, 0.45, 0.92);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    window.addEventListener("resize", this.resize);
    window.addEventListener("pointermove", this.move);
    this.resize();
  }

  async start(page: number): Promise<void> {
    await document.fonts.load("150px Lobster");
    await document.fonts.load("700 46px Fredoka");
    await document.fonts.load("italic 44px 'IM Fell English'");
    this.assets = await loadAssets();
    if (this.disposed) return;
    this.current = page - 1;
    this.target = this.current;
    const first = this.sceneAt(this.current);
    this.theater.setBackdrop(first.backdrop);
    first.setOpenness(0);
    this.rising = 0;
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
    for (const popup of this.cache.values()) popup.dispose();
    this.composer.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  private sceneAt(index: number): PopupScene {
    const cached = this.cache.get(index);
    if (cached) return cached;
    const page = PAGES[index];
    if (!page || !this.assets) throw new Error(`no page at ${index}`);
    const popup = SCENES[page.scene]({ assets: this.assets, page: index + 1 });
    popup.group.visible = false;
    this.scene.add(popup.group);
    this.cache.set(index, popup);
    return popup;
  }

  private stepChange(dt: number, time: number): void {
    const change = this.change;
    if (!change) {
      if (this.rising < 1) {
        this.rising = Math.min(this.rising + dt / 1.7, 1);
        this.sceneAt(this.current).setOpenness(this.rising);
      }
      this.theater.setCurtain(0, time, dt);
      return;
    }
    change.elapsed += dt;
    const frame = changeFrame(change.elapsed);
    this.theater.setCurtain(frame.curtain, time, dt);
    if (!change.swapped) {
      this.sceneAt(change.from).setOpenness(frame.oldOpenness);
      if (frame.swapped) {
        change.swapped = true;
        this.sceneAt(change.from).setOpenness(0);
        const next = this.target;
        change.to = next;
        this.current = next;
        this.theater.setBackdrop(this.sceneAt(next).backdrop);
      }
    } else {
      this.sceneAt(change.to).setOpenness(frame.newOpenness);
    }
    if (frame.done) {
      this.change = undefined;
      this.rising = 1;
    }
  }

  private loop = (): void => {
    if (this.disposed) return;
    this.frame = requestAnimationFrame(this.loop);
    this.clock.update();
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const time = this.clock.getElapsed();
    if (!this.change && this.target !== this.current) this.change = { from: this.current, to: this.current, elapsed: 0, swapped: false };
    this.stepChange(dt, time);
    const active = this.sceneAt(this.current);
    for (const [index, popup] of this.cache) {
      if (index === this.current || index === this.change?.from) popup.update(time, dt);
    }
    this.theater.update(time, active.mood, active.focus);
    this.aim(time);
    this.composer.render(dt);
  };

  private aim(time: number): void {
    const aspect = this.camera.aspect;
    const half = Math.tan((this.camera.fov * Math.PI) / 360);
    const width = aspect < 1 ? 3.4 : 5.6;
    const distance = Math.max(12.5, width / (half * aspect));
    const x = this.pointer.x * 0.8 + Math.sin(time * 0.15) * 0.2;
    const y = 3.4 + distance * 0.06 - this.pointer.y * 0.4;
    const goal = new Vector3(x, y, distance);
    if (this.snapped) this.camera.position.lerp(goal, 0.05);
    else this.camera.position.copy(goal);
    this.snapped = true;
    this.look.set(0, aspect < 1 ? 3.6 : 3.3, 0);
    this.camera.lookAt(this.look);
  }

  private resize = (): void => {
    const { clientWidth, clientHeight } = this.host;
    this.camera.aspect = clientWidth / Math.max(clientHeight, 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(clientWidth, clientHeight);
    this.composer.setSize(clientWidth, clientHeight);
    this.snapped = false;
  };

  private move = (event: PointerEvent): void => {
    this.pointer.set((event.clientX / window.innerWidth) * 2 - 1, (event.clientY / window.innerHeight) * 2 - 1);
  };
}
