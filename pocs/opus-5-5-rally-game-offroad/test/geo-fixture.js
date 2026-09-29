import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { placeOf, createGeo } from '../public/js/core/geo.js';
import { createWorld } from '../public/js/core/sim.js';

const root = fileURLToPath(new URL('../public/geo/', import.meta.url));
const grid = (id, name) => new Int16Array(readFileSync(`${root}${id}/${name}.bin`).buffer.slice(0));

export function loadGeo(id) {
  return createGeo(placeOf(id), grid(id, 'near'), grid(id, 'far'));
}

export function worldFor(def) {
  return createWorld(def, loadGeo(def.id));
}
