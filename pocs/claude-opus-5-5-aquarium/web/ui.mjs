import { RACK_COLORS, MATERIALS, FISH, DECOR, MAX_FISH, MAX_GRASS, byId } from './catalog.mjs';
import { countFish } from './state.mjs';
import { thumbnail } from './materials.mjs';

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (k === 'style') node.style.cssText = v;
    else node.setAttribute(k, v);
  }
  node.append(...children);
  return node;
}

function section(title, ...children) {
  const value = el('span', { class: 'value' });
  const head = el('h2', {}, el('span', {}, title), value);
  return { node: el('section', {}, head, ...children), value };
}

function fishSwatch(f) {
  const stops = f.colors.map((c, i) => `${c} ${(i / f.colors.length) * 100}% ${((i + 1) / f.colors.length) * 100}%`).join(',');
  return el('span', { class: 'fish-dot', style: `background:linear-gradient(90deg,${stops})` });
}

export function buildUI(root, actions) {
  const refs = { colors: new Map(), materials: new Map(), fish: new Map(), decor: new Map() };

  const colors = section('Rack color');
  const colorRow = el('div', { class: 'swatches' });
  for (const c of RACK_COLORS) {
    const b = el('button', { class: 'swatch', title: c.name, 'aria-label': c.name, onclick: () => actions.rackColor(c.id) });
    if (c.hex) b.style.background = c.hex;
    else b.classList.add('natural');
    refs.colors.set(c.id, b);
    colorRow.append(b);
  }
  colors.node.append(colorRow);

  const materials = section('Rack material');
  const matGrid = el('div', { class: 'grid' });
  for (const m of MATERIALS) {
    const b = el('button', { class: 'chip', onclick: () => actions.material(m.id) },
      el('img', { src: thumbnail(m), alt: '' }), el('span', {}, m.name));
    refs.materials.set(m.id, b);
    matGrid.append(b);
  }
  materials.node.append(matGrid);

  const fish = section('Fish');
  const fishList = el('div', { class: 'fish-list' });
  for (const f of FISH) {
    const count = el('span', { class: 'count' }, '0');
    const minus = el('button', { class: 'step', 'aria-label': `Remove ${f.name}`, onclick: () => actions.removeFish(f.id) }, '−');
    const plus = el('button', { class: 'step', 'aria-label': `Add ${f.name}`, onclick: () => actions.addFish(f.id) }, '+');
    const row = el('div', { class: 'fish-row' }, fishSwatch(f), el('span', { class: 'name' }, f.name), minus, count, plus);
    refs.fish.set(f.id, { row, count, minus, plus });
    fishList.append(row);
  }
  const feed = el('button', { class: 'primary', onclick: () => actions.feed() }, 'Feed fish');
  const clear = el('button', { class: 'link', onclick: () => actions.clearFish() }, 'Remove all fish');
  fish.node.append(fishList, el('div', { class: 'actions' }, feed, clear));

  const decor = section('Decorations');
  const grassCount = el('span', { class: 'count' }, '0');
  const grassMinus = el('button', { class: 'step', 'aria-label': 'Less seagrass', onclick: () => actions.grass(-1) }, '−');
  const grassPlus = el('button', { class: 'step', 'aria-label': 'More seagrass', onclick: () => actions.grass(1) }, '+');
  const grassRow = el('div', { class: 'fish-row grass-row' }, el('span', { class: 'fish-dot grass-dot' }), el('span', { class: 'name' }, 'Seagrass'), grassMinus, grassCount, grassPlus);
  const decorGrid = el('div', { class: 'grid' });
  for (const d of DECOR) {
    const b = el('button', { class: 'chip toggle', onclick: () => actions.toggleDecor(d.id) }, el('span', { class: 'tick' }), el('span', {}, d.name));
    refs.decor.set(d.id, b);
    decorGrid.append(b);
  }
  decor.node.append(grassRow, decorGrid);

  const extras = section('Extras');
  const shark = el('button', { class: 'switch', onclick: () => actions.toggleShark() }, el('span', { class: 'knob' }), el('span', {}, 'Small shark'));
  const sound = el('button', { class: 'switch', onclick: () => actions.toggleSound() }, el('span', { class: 'knob' }), el('span', {}, 'Sound'));
  const volume = el('input', { type: 'range', min: '0', max: '1', step: '0.01', value: '0.7', 'aria-label': 'Volume', oninput: e => actions.volume(parseFloat(e.target.value)) });
  extras.node.append(shark, el('div', { class: 'sound-row' }, sound, volume));

  root.append(
    el('header', {}, el('h1', {}, 'Aquarium'), el('p', {}, 'Drag to orbit, scroll to zoom')),
    colors.node, materials.node, fish.node, decor.node, extras.node
  );

  return {
    render(s) {
      for (const [id, b] of refs.colors) b.classList.toggle('on', id === s.rackColor);
      colors.value.textContent = byId(RACK_COLORS, s.rackColor).name;
      for (const [id, b] of refs.materials) b.classList.toggle('on', id === s.material);
      materials.value.textContent = byId(MATERIALS, s.material).name;
      for (const [id, r] of refs.fish) {
        const n = countFish(s, id);
        r.count.textContent = String(n);
        r.row.classList.toggle('on', n > 0);
        r.minus.disabled = n === 0;
        r.plus.disabled = s.fish.length >= MAX_FISH;
      }
      fish.value.textContent = `${s.fish.length} / ${MAX_FISH}`;
      for (const [id, b] of refs.decor) b.classList.toggle('on', s.decor.includes(id));
      decor.value.textContent = `${s.decor.length + (s.grass > 0 ? 1 : 0)} / ${DECOR.length + 1}`;
      grassCount.textContent = String(s.grass);
      grassRow.classList.toggle('on', s.grass > 0);
      grassMinus.disabled = s.grass === 0;
      grassPlus.disabled = s.grass >= MAX_GRASS;
      feed.disabled = s.fish.length === 0;
      shark.classList.toggle('on', s.shark);
      sound.classList.toggle('on', s.sound);
      volume.disabled = !s.sound;
    }
  };
}
