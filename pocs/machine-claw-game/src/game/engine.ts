import * as THREE from "three";
import { createSfx } from "../audio/sfx";
import { logger } from "../logger";
import { initPhysics } from "../physics/arena";
import { buildCabinet } from "../render/cabinet";
import { createCameraRig } from "../render/cameraRig";
import { createClawRig } from "../render/clawRig";
import { createCoinToss, createConfetti } from "../render/effects";
import { createPlush } from "../render/plushMesh";
import { createStage } from "../render/stage";
import { CLAW } from "./layout";
import type { Phase } from "./machine";
import { type SimEvent, createSimulation } from "./simulation";

export type Banner = "insert" | "aim" | "watch" | "won" | "miss";

export type HudState = {
  readonly phase: Phase;
  readonly credits: number;
  readonly seconds: number;
  readonly prizes: number;
  readonly banner: Banner;
  readonly shelf: readonly number[];
};

export type Engine = {
  subscribe(listener: () => void): () => void;
  snapshot(): HudState;
  steer(x: number, z: number): void;
  drop(): void;
  insertCoin(): void;
  toggleView(): void;
  dispose(): void;
};

const RESULT_SECONDS = 2.6;
const MAX_STEP = 1 / 30;

function bannerFor(phase: Phase): Banner {
  if (phase === "idle") return "insert";
  return phase === "aiming" ? "aim" : "watch";
}

export async function createEngine(canvas: HTMLCanvasElement): Promise<Engine> {
  await initPhysics();
  const sim = createSimulation({ random: Math.random });
  const stage = createStage(canvas);
  const cabinet = buildCabinet();
  const rig = createClawRig();
  const confetti = createConfetti();
  const coin = createCoinToss(cabinet.slot);
  const cameraRig = createCameraRig(stage.camera);
  const sfx = createSfx();
  stage.scene.add(cabinet.group, rig.group, confetti.points, coin.mesh);

  const plush = new Map<number, THREE.Group>();
  const listeners = new Set<() => void>();
  const burstAt = new THREE.Vector3(CLAW.homeX, 0.5, CLAW.homeZ);
  let moveX = 0;
  let moveZ = 0;
  let pendingDrop = false;
  let pendingCoin = false;
  let result: Banner | undefined;
  let resultLeft = 0;
  let shelf: readonly number[] = [];
  let hud: HudState = { phase: "idle", credits: 0, seconds: 0, prizes: 0, banner: "insert", shelf };
  let last = performance.now();
  let frameId = 0;

  function react(events: readonly SimEvent[]): void {
    for (const event of events) {
      switch (event.kind) {
        case "phase":
          if (event.phase === "aiming") sfx.start();
          if (event.phase === "descending") sfx.drop();
          if (event.phase === "closing") sfx.grab();
          break;
        case "prize":
          shelf = [...shelf, event.look];
          result = "won";
          resultLeft = RESULT_SECONDS;
          confetti.burst(burstAt);
          sfx.win();
          logger.info("prize won", { look: event.look, prizes: sim.state().prizes });
          break;
        case "roundEnd":
          if (!event.won) {
            result = "miss";
            resultLeft = RESULT_SECONDS;
            sfx.miss();
          }
          break;
        default: {
          const unreachable: never = event;
          return unreachable;
        }
      }
    }
  }

  function syncToys(): void {
    const toys = sim.arena.toys();
    for (const toy of toys) {
      let mesh = plush.get(toy.id);
      if (!mesh) {
        mesh = createPlush(toy.look, toy.radius);
        plush.set(toy.id, mesh);
        stage.scene.add(mesh);
      }
      const at = toy.body.translation();
      const turn = toy.body.rotation();
      mesh.position.set(at.x, at.y, at.z);
      mesh.quaternion.set(turn.x, turn.y, turn.z, turn.w);
    }
    if (plush.size === toys.length) return;
    for (const [id, mesh] of plush) {
      if (toys.some((toy) => toy.id === id)) continue;
      stage.scene.remove(mesh);
      plush.delete(id);
    }
  }

  function publish(): void {
    const state = sim.state();
    const next: HudState = {
      phase: state.phase,
      credits: state.credits,
      seconds: Math.ceil(state.timeLeft),
      prizes: state.prizes,
      banner: result ?? bannerFor(state.phase),
      shelf,
    };
    const changed =
      next.phase !== hud.phase ||
      next.credits !== hud.credits ||
      next.seconds !== hud.seconds ||
      next.prizes !== hud.prizes ||
      next.banner !== hud.banner ||
      next.shelf !== hud.shelf;
    if (!changed) return;
    hud = next;
    for (const listener of listeners) listener();
  }

  function frame(now: number): void {
    const dt = Math.min((now - last) / 1000, MAX_STEP);
    last = now;
    if (dt > 0) {
      react(sim.step({ moveX, moveZ, drop: pendingDrop, coin: pendingCoin }, dt));
      pendingDrop = false;
      pendingCoin = false;
      resultLeft -= dt;
      if (resultLeft <= 0) result = undefined;
      const phase = sim.state().phase;
      const steering = phase === "aiming" && (moveX !== 0 || moveZ !== 0);
      sfx.motor(steering || phase === "descending" || phase === "lifting" || phase === "returning");
      syncToys();
      rig.sync(sim.arena);
      cabinet.animate(now / 1000, moveX, moveZ, phase === "aiming");
      confetti.update(dt);
      coin.update(dt);
      const claw = sim.claw();
      cameraRig.update(dt, claw.x, claw.z);
      stage.render();
      publish();
    }
    frameId = requestAnimationFrame(frame);
  }

  function resize(): void {
    cameraRig.fit(stage.resize());
  }

  resize();
  window.addEventListener("resize", resize);
  frameId = requestAnimationFrame(frame);
  logger.info("engine ready");

  return {
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    snapshot: () => hud,
    steer: (x, z) => {
      const length = Math.max(1, Math.hypot(x, z));
      moveX = x / length;
      moveZ = z / length;
    },
    drop: () => {
      pendingDrop = true;
    },
    insertCoin: () => {
      sfx.unlock();
      sfx.coin();
      coin.toss();
      pendingCoin = true;
      logger.info("coin inserted");
    },
    toggleView: () => cameraRig.toggle(),
    dispose: () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", resize);
      sfx.dispose();
      stage.dispose();
    },
  };
}
