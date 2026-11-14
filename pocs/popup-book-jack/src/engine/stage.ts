import { ACESFilmicToneMapping, Group, HalfFloatType, PCFSoftShadowMap, PerspectiveCamera, SRGBColorSpace, Scene, TextureLoader, Timer, Vector2, Vector3, WebGLRenderTarget, WebGLRenderer } from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { SMAAPass } from "three/addons/postprocessing/SMAAPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { SCENES } from "../scenes";
import type { Assets } from "../scenes/kit";
import { STALK_BASE, STATIONS, type Station } from "../scenes/stations";
import { STALK_HEIGHT, crashed, growth, smooth, stalkFall } from "../scenes/timelines";
import { PAGES } from "../story/pages";
import { Book, halfTexture } from "./book";
import { type Shot, arc, ease, flightShot, flightTime } from "./flight";
import { platform } from "./platform";
import type { PopupScene } from "./popup";
import { StalkWorld } from "./stalkWorld";
import { ARRIVE_TIME, DEPART_TIME, arriveOpenness, bank, departOpenness, leafAt } from "./transition";
import { World } from "./world";

type Phase =
  | { kind: "arrive"; t: number }
  | { kind: "rest" }
  | { kind: "depart"; t: number }
  | { kind: "fly"; from: Shot; origin: number; to: number; elapsed: number; duration: number; turned: boolean };

const BEANSTALK_PAGE = PAGES.findIndex((p) => p.scene === "beanstalk");
const END_PAGE = PAGES.findIndex((p) => p.scene === "theEnd");

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`failed to load ${url}`));
    image.src = url;
  });
}

async function loadAssets(): Promise<Assets> {
  const loader = new TextureLoader();
  const [grass, bricks, spark, disc] = await Promise.all([
    loadImage("assets/grasslight-big.jpg"),
    loader.loadAsync("assets/brick_diffuse.jpg"),
    loader.loadAsync("assets/spark1.png"),
    loader.loadAsync("assets/disc.png"),
  ]);
  bricks.colorSpace = SRGBColorSpace;
  return { grass, bricks, spark, disc };
}

function stationOf(index: number): Station {
  const page = PAGES[index];
  if (!page) throw new Error(`no page at ${index}`);
  return STATIONS[page.scene];
}

export class Stage {
  private readonly renderer: WebGLRenderer;
  private readonly composer: EffectComposer;
  private readonly bloom: UnrealBloomPass;
  private readonly scene = new Scene();
  private readonly camera = new PerspectiveCamera(40, 1, 0.1, 400);
  private readonly clock = new Timer();
  private readonly popups: PopupScene[] = [];
  private readonly pointer = new Vector2();
  private readonly book = new Book();
  private readonly carrier = new Group();
  private stalk?: StalkWorld;
  private world?: World;
  private phase: Phase = { kind: "arrive", t: 0 };
  private night = 0;
  private arrived = 0;
  private target = 0;
  private arrivedAt = -1;
  private quake = 0;
  private wasCrashed = false;
  private frame = 0;
  private disposed = false;

  constructor(private readonly host: HTMLElement, private readonly onReady: () => void) {
    this.renderer = new WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFSoftShadowMap;
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.outputColorSpace = SRGBColorSpace;
    host.appendChild(this.renderer.domElement);
    this.composer = new EffectComposer(this.renderer, new WebGLRenderTarget(1, 1, { samples: 4, type: HalfFloatType }));
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new Vector2(1, 1), 0.3, 0.45, 1.25);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new SMAAPass());
    this.composer.addPass(new OutputPass());
    window.addEventListener("resize", this.resize);
    window.addEventListener("pointermove", this.move);
    this.resize();
  }

  async start(page: number): Promise<void> {
    await document.fonts.load("120px Lobster");
    await document.fonts.load("700 46px Fredoka");
    await document.fonts.load("italic 44px 'IM Fell English'");
    const assets = await loadAssets();
    if (this.disposed) return;
    this.world = new World(this.scene, assets.grass);
    this.build(assets);
    this.arrived = page - 1;
    this.target = this.arrived;
    this.night = stationOf(this.arrived).night ? 1 : 0;
    this.showSpread(this.arrived);
    this.park(this.arrived);
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
    for (const popup of this.popups) popup.dispose();
    this.composer.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  private build(assets: Assets): void {
    const stalk = new StalkWorld(this.scene);
    this.stalk = stalk;
    this.carrier.add(this.book.group);
    PAGES.forEach((page, index) => {
      const station = STATIONS[page.scene];
      const popup = SCENES[page.scene]({ assets, page: index + 1 });
      popup.setOpenness(0);
      this.carrier.add(popup.group);
      this.popups.push(popup);
      const base = platform(station.platform);
      const at = new Vector3(station.at.x, station.at.y, station.at.z);
      if (station.platform === "leaf") stalk.attach(base, at);
      else {
        base.position.copy(at);
        this.scene.add(base);
      }
    });
    this.scene.add(this.carrier);
  }

  private backdrop(index: number) {
    const popup = this.popups[index];
    if (!popup) throw new Error(`no scene at ${index}`);
    return popup.backdrop;
  }

  private showSpread(index: number): void {
    this.book.setLeft(halfTexture(this.backdrop(index), "left"));
    this.book.setRight(halfTexture(this.backdrop(index), "right"));
  }

  private park(index: number): void {
    const { x, y, z } = stationOf(index).at;
    this.carrier.position.set(x, y, z);
    this.carrier.rotation.set(0, 0, 0);
  }

  private since(time: number): number {
    if (this.phase.kind !== "arrive" && this.phase.kind !== "rest") return 0;
    if (this.arrivedAt < 0) return 0;
    return Math.max(time - this.arrivedAt - ARRIVE_TIME, 0);
  }

  private stalkAmount(time: number): number {
    if (this.arrived < BEANSTALK_PAGE) return 0;
    if (this.arrived > BEANSTALK_PAGE) return 1;
    return this.phase.kind === "fly" ? 0 : growth(this.since(time));
  }

  private updateStalk(time: number, dt: number): void {
    const stalk = this.stalk;
    if (!stalk) return;
    const flyingIn = this.phase.kind === "fly" && this.phase.to === BEANSTALK_PAGE;
    const ending = this.arrived === END_PAGE && this.phase.kind !== "fly";
    const since = ending ? this.since(time) : 0;
    const down = ending && crashed(since);
    if (down && !this.wasCrashed) {
      this.quake = 1.2;
      stalk.crash();
    }
    this.wasCrashed = down;
    stalk.update(this.arrived >= BEANSTALK_PAGE || flyingIn, this.stalkAmount(time), stalkFall(since), dt);
  }

  private distanceFactor(): number {
    const aspect = this.camera.aspect;
    return Math.max(1, (aspect < 1 ? 1.3 : 1.55) / aspect);
  }

  private stationShot(index: number, time: number): Shot {
    const station = stationOf(index);
    const popup = this.popups[index];
    const d = this.distanceFactor();
    const { x, y, z } = station.at;
    const shot: Shot = { pos: { x, y: y + 5.4 * d, z: z + 10.4 * d }, look: { x, y: y + (popup?.lookHeight ?? 1), z: z - 0.1 } };
    if (index !== BEANSTALK_PAGE || index !== this.arrived || this.phase.kind === "fly") return shot;
    const amount = this.stalkAmount(time);
    const tip = amount * STALK_HEIGHT;
    const rise = smooth(amount, 0.02, 0.3);
    shot.pos.x += (STALK_BASE.x - 2 - x) * rise;
    shot.pos.y = Math.max(shot.pos.y, tip + 3);
    shot.pos.z += (STALK_BASE.z + 17 - shot.pos.z) * rise;
    shot.look.x += (STALK_BASE.x - x) * rise;
    shot.look.y = Math.max(shot.look.y, tip - 1.5);
    shot.look.z += (STALK_BASE.z - z) * rise;
    return shot;
  }

  private takeOff(from: Shot, origin: number, time: number): Phase {
    const to = this.target;
    const forward = to > origin;
    const leaving = this.backdrop(origin);
    const coming = this.backdrop(to);
    if (forward) {
      this.book.leaf.setFaces(halfTexture(leaving, "right"), halfTexture(coming, "mirroredLeft"));
      this.book.setRight(halfTexture(coming, "right"));
    } else {
      this.book.leaf.setFaces(halfTexture(coming, "right"), halfTexture(leaving, "mirroredLeft"));
      this.book.setLeft(halfTexture(coming, "left"));
    }
    return { kind: "fly", from, origin, to, elapsed: 0, duration: flightTime(from, this.stationShot(to, time)) + 0.6, turned: false };
  }

  private carry(now: Extract<Phase, { kind: "fly" }>, progress: number): void {
    const a = stationOf(now.origin).at;
    const b = stationOf(now.to).at;
    const t = ease(progress);
    const lift = Math.sin(Math.PI * t) * arc(now.from, this.stationShot(now.to, 0));
    const sway = bank(progress, b.x >= a.x ? -1 : 1);
    this.carrier.position.set(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t + lift + sway.lift, a.z + (b.z - a.z) * t);
    this.carrier.rotation.set(sway.pitch, 0, sway.roll);
    const forward = now.to > now.origin;
    const leaf = leafAt(progress);
    this.book.leaf.bend(forward ? leaf : 1 - leaf);
    if (leaf >= 1 && !now.turned) {
      now.turned = true;
      this.showSpread(now.to);
    }
  }

  private advance(dt: number, time: number): Shot {
    const popup = this.popups[this.arrived];
    if (this.phase.kind === "rest" && this.target !== this.arrived) this.phase = { kind: "depart", t: 0 };
    if (this.phase.kind === "arrive" && this.target !== this.arrived) this.phase = { kind: "depart", t: (1 - (popup?.openness ?? 0)) * DEPART_TIME };
    const now = this.phase;
    if (now.kind === "rest") {
      popup?.setOpenness(1);
      return this.stationShot(this.arrived, time);
    }
    if (now.kind === "arrive") {
      if (this.arrivedAt < 0) this.arrivedAt = time;
      now.t += dt;
      popup?.setOpenness(arriveOpenness(now.t));
      if (now.t >= ARRIVE_TIME) this.phase = { kind: "rest" };
      return this.stationShot(this.arrived, time);
    }
    if (now.kind === "depart") {
      now.t += dt;
      popup?.setOpenness(departOpenness(now.t));
      const shot = this.stationShot(this.arrived, time);
      if (now.t >= DEPART_TIME) this.phase = this.takeOff(shot, this.arrived, time);
      return shot;
    }
    now.elapsed += dt;
    const progress = Math.min(now.elapsed / now.duration, 1);
    const shot = flightShot(now.from, this.stationShot(now.to, time), progress);
    this.carry(now, progress);
    if (progress >= 1) {
      this.arrived = now.to;
      this.arrivedAt = time;
      this.park(now.to);
      this.book.leaf.bend(0);
      this.showSpread(now.to);
      this.phase = this.target === now.to ? { kind: "arrive", t: 0 } : this.takeOff(shot, now.to, time);
    }
    return shot;
  }

  private loop = (): void => {
    if (this.disposed) return;
    this.frame = requestAnimationFrame(this.loop);
    this.clock.update();
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const time = this.clock.getElapsed();
    const shot = this.advance(dt, time);
    this.popups.forEach((popup, index) => {
      popup.since = index === this.arrived ? this.since(time) : 0;
      if (popup.openness > 0) popup.update(time, dt);
    });
    this.updateStalk(time, dt);
    const heading = this.phase.kind === "fly" ? this.phase.to : this.arrived;
    const goalNight = stationOf(heading).night ? 1 : 0;
    this.night += (goalNight - this.night) * Math.min(dt * 1.5, 1);
    const shake = Math.max(this.quake, this.popups[this.arrived]?.shake ?? 0);
    this.quake = Math.max(this.quake - dt * 0.9, 0);
    const jitter = shake * 0.22;
    this.camera.position.set(
      shot.pos.x + this.pointer.x * 1.0 + Math.sin(time * 0.15) * 0.25 + Math.sin(time * 53) * jitter,
      shot.pos.y - this.pointer.y * 0.7 + Math.sin(time * 61) * jitter,
      shot.pos.z,
    );
    this.camera.lookAt(shot.look.x, shot.look.y, shot.look.z);
    this.world?.update(shot.look, shot.pos.y, this.night, time);
    this.composer.render(dt);
  };

  private resize = (): void => {
    const { clientWidth, clientHeight } = this.host;
    this.camera.aspect = clientWidth / Math.max(clientHeight, 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(clientWidth, clientHeight);
    this.composer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.composer.setSize(clientWidth, clientHeight);
    this.bloom.resolution.set(clientWidth, clientHeight);
  };

  private move = (event: PointerEvent): void => {
    this.pointer.set((event.clientX / window.innerWidth) * 2 - 1, (event.clientY / window.innerHeight) * 2 - 1);
  };
}
