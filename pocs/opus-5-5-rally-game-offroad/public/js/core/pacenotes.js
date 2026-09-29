import { G } from './vehicle.js';

const CALLS = [[45, 1], [60, 2], [80, 3], [110, 4], [160, 5]];
const STRAIGHT = 450;
const MARGIN = 0.82;

function severity(radius) {
  for (const [r, s] of CALLS) if (radius < r) return s;
  return 6;
}

export function paceNotes(track, grip = 0.8) {
  const notes = [];
  const n = track.count;
  let i = 0;
  while (i < n && Math.abs(track.curvature[i]) > 1 / STRAIGHT) i++;
  const start = i;
  let k = 0;
  while (k < n) {
    const j = (start + k) % n;
    if (Math.abs(track.curvature[j]) <= 1 / STRAIGHT) {
      k++;
      continue;
    }
    const sign = Math.sign(track.curvature[j]);
    let peak = 0;
    let len = 0;
    while (k < n) {
      const m = (start + k) % n;
      const c = track.curvature[m];
      if (Math.abs(c) <= 1 / STRAIGHT || Math.sign(c) !== sign) break;
      peak = Math.max(peak, Math.abs(c));
      len++;
      k++;
    }
    const radius = 1 / peak;
    notes.push({ i: j, length: len * track.spacing, dir: sign > 0 ? 'LEFT' : 'RIGHT', radius, severity: severity(radius), speed: Math.sqrt(grip * G * radius) * MARGIN });
  }
  return notes.sort((a, b) => a.i - b.i);
}

export function nextNote(track, notes, i) {
  let best = null;
  let bestDist = Infinity;
  for (const note of notes) {
    const ahead = (((note.i - i) % track.count) + track.count) % track.count * track.spacing;
    const inside = ahead === 0 || track.length - ahead <= note.length;
    const d = inside ? 0 : ahead;
    if (d < bestDist) {
      bestDist = d;
      best = note;
    }
  }
  return best ? { note: best, distance: bestDist } : null;
}
