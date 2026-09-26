import { RACK_COLORS, MATERIALS, FISH, DECOR, MAX_FISH, MAX_GRASS, byId } from './catalog.mjs';

export function createState() {
  return {
    rackColor: 'natural',
    material: 'wood',
    fish: ['neon', 'neon', 'neon', 'neon', 'clown', 'clown', 'bluetang', 'yellowtang'],
    grass: 3,
    decor: ['rocks', 'ship'],
    shark: false,
    sound: false
  };
}

function need(list, id, what) {
  if (!byId(list, id)) throw new Error(`unknown ${what} ${id}`);
}

export function setRackColor(s, id) {
  need(RACK_COLORS, id, 'rack color');
  return { ...s, rackColor: id };
}

export function setMaterial(s, id) {
  need(MATERIALS, id, 'material');
  return { ...s, material: id };
}

export function addFish(s, id) {
  need(FISH, id, 'fish');
  if (s.fish.length >= MAX_FISH) return s;
  return { ...s, fish: [...s.fish, id] };
}

export function removeFish(s, id) {
  const i = s.fish.lastIndexOf(id);
  if (i < 0) return s;
  return { ...s, fish: s.fish.filter((_, j) => j !== i) };
}

export function clearFish(s) {
  return { ...s, fish: [] };
}

export function countFish(s, id) {
  return s.fish.filter(f => f === id).length;
}

export function toggleDecor(s, id) {
  need(DECOR, id, 'decoration');
  const on = s.decor.includes(id);
  return { ...s, decor: on ? s.decor.filter(d => d !== id) : [...s.decor, id] };
}

export function setGrass(s, level) {
  const next = Math.max(0, Math.min(MAX_GRASS, Math.round(level)));
  return next === s.grass ? s : { ...s, grass: next };
}

export function toggleShark(s) {
  return { ...s, shark: !s.shark };
}

export function toggleSound(s) {
  return { ...s, sound: !s.sound };
}

export function restore(saved) {
  const base = createState();
  if (!saved || typeof saved !== 'object') return base;
  return {
    rackColor: byId(RACK_COLORS, saved.rackColor) ? saved.rackColor : base.rackColor,
    material: byId(MATERIALS, saved.material) ? saved.material : base.material,
    fish: Array.isArray(saved.fish) ? saved.fish.filter(f => byId(FISH, f)).slice(0, MAX_FISH) : base.fish,
    grass: Number.isInteger(saved.grass) ? setGrass(base, saved.grass).grass : base.grass,
    decor: Array.isArray(saved.decor) ? [...new Set(saved.decor.filter(d => byId(DECOR, d)))] : base.decor,
    shark: saved.shark === true,
    sound: false
  };
}
