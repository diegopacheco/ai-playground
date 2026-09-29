import { clamp } from './core/math.js';

const THROTTLE = ['KeyW', 'ArrowUp'];
const BRAKE = ['KeyS', 'ArrowDown'];
const LEFT = ['KeyA', 'ArrowLeft'];
const RIGHT = ['KeyD', 'ArrowRight'];
const GAME_KEYS = new Set([...THROTTLE, ...BRAKE, ...LEFT, ...RIGHT, 'Space']);

export function createInput(onAction) {
  const down = new Set();
  const state = { throttle: 0, brake: 0, steer: 0, handbrake: false };
  addEventListener('keydown', (e) => {
    if (GAME_KEYS.has(e.code)) e.preventDefault();
    if (!e.repeat) onAction(e.code);
    down.add(e.code);
  });
  addEventListener('keyup', (e) => down.delete(e.code));
  addEventListener('blur', () => down.clear());
  const any = (keys) => keys.some((k) => down.has(k));
  return {
    read(dt, speed) {
      const target = (any(RIGHT) ? 1 : 0) - (any(LEFT) ? 1 : 0);
      const rate = target === 0 ? 5 : Math.sign(target) !== Math.sign(state.steer) ? 7 : 3.2 - Math.min(1.6, speed / 25);
      state.steer += clamp(target - state.steer, -rate * dt, rate * dt);
      state.throttle += clamp((any(THROTTLE) ? 1 : 0) - state.throttle, -dt * 8, dt * 5);
      state.brake += clamp((any(BRAKE) ? 1 : 0) - state.brake, -dt * 10, dt * 6);
      state.handbrake = down.has('Space');
      return state;
    },
    clear() {
      down.clear();
      Object.assign(state, { throttle: 0, brake: 0, steer: 0, handbrake: false });
    },
  };
}
