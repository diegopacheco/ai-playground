import { createControls, rampControls } from './core/controls.js';

const THROTTLE = ['KeyW', 'ArrowUp'];
const BRAKE = ['KeyS', 'ArrowDown'];
const LEFT = ['KeyA', 'ArrowLeft'];
const RIGHT = ['KeyD', 'ArrowRight'];
const GAME_KEYS = new Set([...THROTTLE, ...BRAKE, ...LEFT, ...RIGHT, 'Space']);

export function createInput(onAction) {
  const down = new Set();
  const state = createControls();
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
      return rampControls(state, { throttle: any(THROTTLE), brake: any(BRAKE), left: any(LEFT), right: any(RIGHT), handbrake: down.has('Space'), lookBack: down.has('KeyB') }, dt, speed);
    },
    clear() {
      down.clear();
      Object.assign(state, createControls());
    },
  };
}
