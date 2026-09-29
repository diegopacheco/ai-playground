import { clamp } from './math.js';

export function createControls() {
  return { throttle: 0, brake: 0, steer: 0, handbrake: false, lookBack: false };
}

export function rampControls(state, keys, dt, speed) {
  const target = (keys.right ? 1 : 0) - (keys.left ? 1 : 0);
  const rate = target === 0 ? 6 : Math.sign(target) !== Math.sign(state.steer) ? 9 : 5 - Math.min(1.5, speed / 30);
  state.steer += clamp(target - state.steer, -rate * dt, rate * dt);
  state.throttle += clamp((keys.throttle ? 1 : 0) - state.throttle, -dt * 8, dt * 5);
  state.brake += clamp((keys.brake ? 1 : 0) - state.brake, -dt * 10, dt * 6);
  state.handbrake = !!keys.handbrake;
  state.lookBack = !!keys.lookBack;
  return state;
}
