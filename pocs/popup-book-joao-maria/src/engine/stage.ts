import {
  ACESFilmicToneMapping,
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  Color,
  DirectionalLight,
  FogExp2,
  HemisphereLight,
  Mesh,
  MeshStandardMaterial,
  PCFShadowMap,
  PerspectiveCamera,
  PlaneGeometry,
  Points,
  PointsMaterial,
  RepeatWrapping,
  SRGBColorSpace,
  Scene,
  type Texture,
  TextureLoader,
  Timer,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { SCENES } from "../scenes";
import type { Assets } from "../scenes/kit";
import { PAGES } from "../story/pages";
import { Book, halfTexture } from "./book";
import type { PopupScene } from "./popup";
import { FOLD_TIME, TOTAL_TURN_TIME, TURN_TIME, turnFrame } from "./timeline";

interface Turn {
  from: number;
  to: number;
  elapsed: number;
  leafStarted: boolean;
  landed: boolean;
}

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

export class Stage {
  private readonly renderer: WebGLRenderer;
  private readonly scene = new Scene();
  private readonly camera = new PerspectiveCamera(36, 1, 0.1, 100);
  private readonly book = new Book();
  private readonly clock = new Timer();
  private readonly cache = new Map<number, PopupScene>();
  private readonly pointer = new Vector2();
  private readonly look = new Vector3(0, 1.0, -0.1);
  private assets?: Assets;
  private current = 0;
  private target = 0;
  private turn?: Turn;
  private dust?: Points;
  private frame = 0;
  private disposed = false;

  constructor(private readonly host: HTMLElement, private readonly onReady: () => void) {
    this.renderer = new WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFShadowMap;
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.renderer.outputColorSpace = SRGBColorSpace;
    host.appendChild(this.renderer.domElement);
    this.scene.background = new Color("#1b100b");
    this.scene.fog = new FogExp2("#1b100b", 0.035);
    this.scene.add(this.book.group);
    this.lights();
    window.addEventListener("resize", this.resize);
    window.addEventListener("pointermove", this.move);
    this.resize();
  }

  async start(page: number): Promise<void> {
    await document.fonts.load("120px Lobster");
    await document.fonts.load("700 46px Fredoka");
    await document.fonts.load("italic 44px 'IM Fell English'");
    this.assets = await loadAssets();
    if (this.disposed) return;
    this.table();
    this.dust = this.motes(this.assets.disc);
    this.current = page - 1;
    this.target = this.current;
    const first = this.sceneAt(this.current);
    this.showSpread(first);
    first.setOpenness(0);
    this.turn = { from: this.current, to: this.current, elapsed: FOLD_TIME + TURN_TIME, leafStarted: true, landed: true };
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
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  private lights(): void {
    this.scene.add(new HemisphereLight("#ffe9c7", "#3a2414", 1.1));
    const sun = new DirectionalLight("#fff1d6", 2.6);
    sun.position.set(-5, 11, 7);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -7;
    sun.shadow.camera.right = 7;
    sun.shadow.camera.top = 7;
    sun.shadow.camera.bottom = -7;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    sun.shadow.radius = 4;
    this.scene.add(sun);
    const rim = new DirectionalLight("#9fc4ff", 0.6);
    rim.position.set(6, 4, -6);
    this.scene.add(rim);
  }

  private table(): void {
    const loader = new TextureLoader();
    const wood: Texture = loader.load("assets/hardwood2_diffuse.jpg");
    wood.colorSpace = SRGBColorSpace;
    wood.wrapS = RepeatWrapping;
    wood.wrapT = RepeatWrapping;
    wood.repeat.set(4, 4);
    wood.anisotropy = 8;
    const mesh = new Mesh(new PlaneGeometry(40, 40), new MeshStandardMaterial({ map: wood, roughness: 0.55, color: "#c9a07a" }));
    mesh.rotation.x = -Math.PI / 2;
    mesh.receiveShadow = true;
    this.scene.add(mesh);
  }

  private motes(texture: Texture): Points {
    const count = 260;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 12;
      positions[i * 3 + 1] = Math.random() * 5;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 9;
    }
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(positions, 3));
    const points = new Points(geometry, new PointsMaterial({ map: texture, size: 0.06, color: "#ffe2a8", transparent: true, opacity: 0.7, depthWrite: false, blending: AdditiveBlending }));
    this.scene.add(points);
    return points;
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

  private showSpread(popup: PopupScene): void {
    this.book.setLeft(halfTexture(popup.backdrop, "left"));
    this.book.setRight(halfTexture(popup.backdrop, "right"));
  }

  private beginTurn(): void {
    const step = this.target > this.current ? 1 : -1;
    this.turn = { from: this.current, to: this.current + step, elapsed: 0, leafStarted: false, landed: false };
    this.sceneAt(this.turn.to);
  }

  private stepTurn(dt: number): void {
    const turn = this.turn;
    if (!turn) return;
    turn.elapsed += dt;
    const frame = turnFrame(turn.elapsed);
    const from = this.sceneAt(turn.from);
    const to = this.sceneAt(turn.to);
    const forward = turn.to > turn.from;
    if (turn.from !== turn.to) from.setOpenness(frame.oldOpenness);
    if (frame.leaf > 0 && !turn.leafStarted && turn.from !== turn.to) {
      turn.leafStarted = true;
      from.group.visible = false;
      if (forward) {
        this.book.leaf.setFaces(halfTexture(from.backdrop, "right"), halfTexture(to.backdrop, "mirroredLeft"));
        this.book.setRight(halfTexture(to.backdrop, "right"));
      } else {
        this.book.leaf.setFaces(halfTexture(to.backdrop, "right"), halfTexture(from.backdrop, "mirroredLeft"));
        this.book.setLeft(halfTexture(to.backdrop, "left"));
      }
      this.book.leaf.group.visible = true;
    }
    if (turn.leafStarted && !turn.landed) {
      this.book.leaf.bend(forward ? frame.leaf : 1 - frame.leaf);
      if (frame.leaf >= 1) {
        turn.landed = true;
        this.book.leaf.group.visible = false;
        this.showSpread(to);
        this.current = turn.to;
      }
    }
    if (turn.landed) to.setOpenness(frame.newOpenness);
    if (turn.elapsed >= TOTAL_TURN_TIME) this.turn = undefined;
  }

  private loop = (): void => {
    if (this.disposed) return;
    this.frame = requestAnimationFrame(this.loop);
    this.clock.update();
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const time = this.clock.getElapsed();
    if (!this.turn && this.target !== this.current) this.beginTurn();
    this.stepTurn(dt);
    for (const [index, popup] of this.cache) {
      if (index === this.current || index === this.turn?.from || index === this.turn?.to) popup.update(time, dt);
    }
    if (this.dust) {
      this.dust.rotation.y = time * 0.02;
      this.dust.position.y = Math.sin(time * 0.3) * 0.2;
    }
    this.aim(time);
    this.renderer.render(this.scene, this.camera);
  };

  private aim(time: number): void {
    const aspect = this.camera.aspect;
    const distance = Math.max(1, (aspect < 1 ? 1.3 : 1.55) / aspect);
    if (this.scene.fog instanceof FogExp2) this.scene.fog.density = 0.035 / distance;
    const x = this.pointer.x * 1.2 + Math.sin(time * 0.15) * 0.25;
    const y = 7.6 * distance - this.pointer.y * 0.8;
    const z = 9.0 * distance;
    this.camera.position.lerp(new Vector3(x, y, z), 0.04);
    this.camera.lookAt(this.look);
  }

  private resize = (): void => {
    const { clientWidth, clientHeight } = this.host;
    this.camera.aspect = clientWidth / Math.max(clientHeight, 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(clientWidth, clientHeight);
    if (this.camera.position.lengthSq() === 0) this.camera.position.set(0, 8, 9.6);
  };

  private move = (event: PointerEvent): void => {
    this.pointer.set((event.clientX / window.innerWidth) * 2 - 1, (event.clientY / window.innerHeight) * 2 - 1);
  };
}
