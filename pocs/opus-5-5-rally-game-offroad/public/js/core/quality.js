export const LEVELS = [
  { name: 'Ultra', scale: 1.0, shadows: 4096, particles: 1.0, grass: 1.0 },
  { name: 'High', scale: 0.85, shadows: 2048, particles: 0.8, grass: 0.8 },
  { name: 'Medium', scale: 0.7, shadows: 1024, particles: 0.6, grass: 0.5 },
  { name: 'Low', scale: 0.55, shadows: 1024, particles: 0.4, grass: 0.25 },
  { name: 'Potato', scale: 0.42, shadows: 0, particles: 0.25, grass: 0 },
];

export const TARGET_FPS = 30;
const DOWN_BELOW = 36;
const UP_ABOVE = 58;
const UP_WINDOWS = 4;
const COOLDOWN = 20;

export function createGovernor(level = 1) {
  return { level, frames: 0, elapsed: 0, fps: 60, goodWindows: 0, cooldown: 0, failures: 0, settling: false, changed: false };
}

export function tickGovernor(gov, dt) {
  gov.changed = false;
  gov.frames++;
  gov.elapsed += dt;
  gov.cooldown = Math.max(0, gov.cooldown - dt);
  if (gov.elapsed < 1) return gov;
  gov.fps = gov.frames / gov.elapsed;
  gov.frames = 0;
  gov.elapsed = 0;
  if (gov.settling) {
    gov.settling = false;
    return gov;
  }
  if (gov.fps < DOWN_BELOW && gov.level < LEVELS.length - 1) {
    gov.level++;
    gov.goodWindows = 0;
    gov.cooldown = COOLDOWN * 2 ** gov.failures;
    gov.failures++;
    gov.settling = true;
    gov.changed = true;
  } else if (gov.fps > UP_ABOVE && gov.cooldown === 0) {
    gov.goodWindows++;
    if (gov.goodWindows >= UP_WINDOWS && gov.level > 0) {
      gov.level--;
      gov.goodWindows = 0;
      gov.settling = true;
      gov.changed = true;
    }
  } else {
    gov.goodWindows = 0;
  }
  return gov;
}
