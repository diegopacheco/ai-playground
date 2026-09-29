import { createControls, rampControls } from '../public/js/core/controls.js';
import { nextNote } from '../public/js/core/pacenotes.js';
import { rightAt } from '../public/js/core/tracks.js';
import { wrapAngle } from '../public/js/core/math.js';

export function createKeyboardDriver(track, notes, reaction = 0.15) {
  const controls = createControls();
  const queue = [];
  let clock = 0;
  return (car, dt) => {
    clock += dt;
    const look = Math.round((10 + Math.max(car.u, 0) * 0.5) / track.spacing);
    const j = (car.proj.i + look) % track.count;
    const [rx, rz] = rightAt(track, j);
    const err = wrapAngle(Math.atan2(track.xs[j] + rx * 0 - car.x, track.zs[j] + rz * 0 - car.z) - car.heading);
    const ahead = nextNote(track, notes, car.proj.i);
    let braking = false;
    if (ahead) {
      const v = ahead.note.speed;
      const need = (car.u * car.u - v * v) / (2 * 5) + 15;
      braking = car.u > v + 1 && ahead.distance < need;
    }
    queue.push({ at: clock + reaction, keys: { left: err > 0.05, right: err < -0.05, throttle: !braking, brake: braking } });
    let keys = { left: false, right: false, throttle: false, brake: false };
    while (queue.length && queue[0].at <= clock) keys = queue.shift().keys;
    return rampControls(controls, keys, dt, Math.abs(car.u));
  };
}
