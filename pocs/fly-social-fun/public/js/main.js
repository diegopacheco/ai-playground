import { createStage } from './stage.js';
import { createFlock } from './flock.js';
import { createBubbles } from './bubbles.js';
import { createPanel } from './panel.js';
import { createProfile } from './profile.js';
import { connect, postAction, fetchProfile } from './api.js';

const SWAT_IMPACT_MS = 430;
const DEATH_WORDS = { swatter: 'SMACK!', spider: 'CHOMP', age: 'x_x' };
const MODES = {
  swat: { allowed: () => true, hint: 'Swatter armed: click any spot in the kitchen' },
  snack: { allowed: (spot) => spot.kind === 'food', hint: 'Snack ready: click the banana, pizza or trash can' },
};

const stageEl = document.getElementById('stage');
const hintEl = document.getElementById('hint');
const toastEl = document.getElementById('toast');
const connectionEl = document.getElementById('connection');
const defaultHint = hintEl.textContent;

const stage = createStage(stageEl);
const flock = createFlock(stage);
const bubbles = createBubbles(document.getElementById('bubbles'), stage.camera, flock.position);
let world = { spots: [], species: {}, tickMs: 1500 };
let mode = null;

const profile = createProfile(document.getElementById('profile'), {
  load: fetchProfile,
  spotLabel: (spotId) => world.spots.find((spot) => spot.id === spotId)?.label || spotId,
  speciesOf: (speciesId) => world.species[speciesId],
  onClose: () => flock.select(null),
});

const panel = createPanel(document.querySelector('.panel'), { onSelectFly: selectFly });

function selectFly(flyId) {
  const position = flock.position(flyId);
  if (position) stage.focus(position);
  flock.select(flyId);
  profile.open(flyId);
}

function toast(message) {
  toastEl.textContent = message;
  toastEl.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (toastEl.hidden = true), 2600);
}

function setMode(next) {
  mode = mode === next ? null : next;
  document.querySelectorAll('[data-mode]').forEach((button) => button.classList.toggle('armed', button.dataset.mode === mode));
  stageEl.classList.toggle('aiming', Boolean(mode));
  hintEl.textContent = mode ? MODES[mode].hint : defaultHint;
}

const handlers = {
  hello(event) {
    world = event;
    stage.buildSpots(event.spots);
    flock.sync(event.flies, event.species);
    panel.reset(event);
  },
  buzz({ buzz }) {
    panel.addBuzz(buzz);
    bubbles.show(buzz);
  },
  move({ flyId, spot }) {
    flock.moveTo(flyId, spot);
  },
  like({ flyId, authorId, buzzId, likes }) {
    flock.heartTrail(flyId, authorId);
    panel.updateLikes(buzzId, likes);
  },
  follow({ flyId, targetId }) {
    flock.followLink(flyId, targetId);
  },
  hatch({ fly }) {
    flock.add(fly);
  },
  death({ flyId, cause }) {
    const die = () => {
      flock.kill(flyId, cause);
      const position = flock.position(flyId);
      if (position && cause !== 'swatter') bubbles.burst(position, DEATH_WORDS[cause]);
    };
    if (cause === 'swatter') setTimeout(die, SWAT_IMPACT_MS);
    else die();
  },
  swat({ spot }) {
    stage.swat(spot);
    setTimeout(() => bubbles.burst(stage.spotPosition(spot), DEATH_WORDS.swatter), SWAT_IMPACT_MS);
  },
  snack({ spot, ticks }) {
    stage.snack(spot, (ticks * world.tickMs) / 1000);
    bubbles.burst(stage.spotPosition(spot), 'SNACK!');
  },
  tick(event) {
    panel.update(event);
    profile.refresh();
  },
};

connect(
  (event) => handlers[event.type]?.(event),
  (status) => {
    connectionEl.textContent = status;
    connectionEl.classList.toggle('live', status === 'live');
  },
);

stage.onFrame((dt, time) => {
  flock.update(dt, time);
  bubbles.update();
});

stage.onClick(async (event) => {
  if (!mode) {
    const flyId = flock.pick(stage.rayFrom(event));
    if (flyId) selectFly(flyId);
    return;
  }
  const action = mode;
  const spotId = stage.pickSpot(event, MODES[action].allowed);
  setMode(action);
  if (!spotId) return toast(action === 'swat' ? 'Missed the whole kitchen. Impressive.' : 'Snacks only go on food.');
  try {
    const result = await postAction(action, spotId);
    if (action === 'swat') toast(result.killed ? `Swatted ${result.killed} and ${result.survived} got away` : 'Nobody was there. The flies are laughing.');
    else toast(`Dropped ${result.snack} on the ${spotId}`);
  } catch (error) {
    toast(error.message);
  }
});

document.querySelectorAll('[data-mode]').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && mode) setMode(mode);
});
