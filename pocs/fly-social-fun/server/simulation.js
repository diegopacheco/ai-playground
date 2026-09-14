import { createRandom } from './random.js';
import { SPOTS, SPECIES, NAMES, BIOS, SNACKS, TICKS_PER_DAY } from './world.js';
import { chooseAction, chooseDestination, resolveSwat, isDueToDie, WEB_CATCH_CHANCE } from './behavior.js';
import { writeBuzz, spotBuzz } from './buzzes.js';
import { extractHashtags, trendingHashtags, rankFlies } from './social.js';

export class SimulationError extends Error {}

const SPOT_BY_ID = new Map(SPOTS.map((spot) => [spot.id, spot]));
const SAFE_SPOTS = SPOTS.filter((spot) => spot.kind !== 'hazard').map((spot) => spot.id);
const MILESTONES = new Set([3, 5, 10, 15, 20, 25]);
const NAME_SUFFIXES = ['Jr.', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'];
const SNACK_TICKS = 20;
const EGG_TICKS = 10;
const MAX_CLUTCH = 3;
const WEB_RESOLVE_TICKS = 2;
const MEMORIAM_SIZE = 12;

const DEFAULTS = {
  seed: Date.now(),
  startFlies: 14,
  minFlies: 8,
  maxFlies: 26,
  feedLimit: 200,
  actChance: 0.3,
};

export function createSimulation(options = {}) {
  const config = { ...DEFAULTS, ...options };
  const rng = createRandom(config.seed);
  const flies = new Map();
  const buzzes = [];
  const snacks = {};
  const eggs = [];
  const nameUses = new Map();
  const listeners = new Set();
  const stats = { buzzes: 0, hatched: 0, deaths: 0 };
  let tickCount = 0;
  let nextFlyId = 1;
  let nextBuzzId = 1;

  const day = (ticks) => Math.floor(ticks / TICKS_PER_DAY);
  const alive = () => [...flies.values()].filter((fly) => fly.alive);
  const at = (spotId) => alive().filter((fly) => fly.spot === spotId);

  function emit(event) {
    for (const listener of listeners) listener(event);
  }

  function subscribe(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  function pickName() {
    const unused = NAMES.filter(([name]) => !nameUses.has(name));
    const [name, handle] = unused.length ? rng.pick(unused) : rng.pick(NAMES);
    const uses = (nameUses.get(name) || 0) + 1;
    nameUses.set(name, uses);
    if (uses === 1) return { name, handle };
    return { name: `${name} ${NAME_SUFFIXES[uses - 2] || uses}`, handle: `${handle}_${uses}` };
  }

  function createFly(spotId, parents = []) {
    const speciesId = parents.length ? rng.pick(parents).species : rng.pick(Object.keys(SPECIES));
    const species = SPECIES[speciesId];
    const fly = {
      id: `fly-${nextFlyId++}`,
      ...pickName(),
      species: speciesId,
      bio: rng.pick(BIOS),
      traits: { chatty: rng.next(), social: rng.next(), reckless: rng.next(), romantic: rng.next() },
      hunger: rng.next() * 0.5,
      hungerRate: 0.01 + rng.next() * 0.02,
      age: 0,
      lifespan: Math.round(species.lifespanDays * TICKS_PER_DAY * (0.8 + rng.next() * 0.4)),
      spot: spotId,
      arrivedAt: tickCount,
      followers: new Set(),
      following: new Set(),
      crushId: null,
      likes: 0,
      buzzCount: 0,
      alive: true,
      cause: null,
      diedAt: null,
      parents: parents.map((parent) => parent.handle),
    };
    flies.set(fly.id, fly);
    return fly;
  }

  function publicFly(fly) {
    return {
      id: fly.id,
      name: fly.name,
      handle: fly.handle,
      species: fly.species,
      speciesLabel: SPECIES[fly.species].label,
      bio: fly.bio,
      spot: fly.spot,
      alive: fly.alive,
      cause: fly.cause,
      ageDays: day(fly.age),
      lifespanDays: day(fly.lifespan),
      hunger: Math.round(fly.hunger * 100) / 100,
      followers: fly.followers.size,
      following: fly.following.size,
      likes: fly.likes,
      buzzCount: fly.buzzCount,
      parents: fly.parents,
    };
  }

  function publicBuzz(buzz) {
    const { likedBy, ...rest } = buzz;
    return rest;
  }

  function post(fly, text, replyTo = null) {
    const buzz = {
      id: `buzz-${nextBuzzId++}`,
      flyId: fly.id,
      name: fly.name,
      handle: fly.handle,
      species: fly.species,
      text,
      hashtags: extractHashtags(text),
      likes: 0,
      likedBy: new Set(),
      replyTo,
      tick: tickCount,
      day: day(tickCount),
    };
    buzzes.unshift(buzz);
    if (buzzes.length > config.feedLimit) buzzes.length = config.feedLimit;
    fly.buzzCount++;
    stats.buzzes++;
    emit({ type: 'buzz', buzz: publicBuzz(buzz) });
    return buzz;
  }

  function moveFly(fly, spotId) {
    fly.spot = spotId;
    fly.arrivedAt = tickCount;
    emit({ type: 'move', flyId: fly.id, spot: spotId });
  }

  function hatch(spotId, parents = []) {
    const fly = createFly(spotId, parents);
    stats.hatched++;
    emit({ type: 'hatch', fly: publicFly(fly) });
    const livingParent = parents.find((parent) => parent.alive);
    if (livingParent) post(fly, writeBuzz('hatchWithParents', rng, { parent: livingParent.handle }));
    else post(fly, writeBuzz('hatch', rng));
    return fly;
  }

  function pruneMemoriam() {
    const dead = [...flies.values()].filter((fly) => !fly.alive).sort((a, b) => b.diedAt - a.diedAt);
    for (const fly of dead.slice(MEMORIAM_SIZE)) flies.delete(fly.id);
  }

  function mourn(victim) {
    const living = alive();
    const followers = living.filter((fly) => victim.followers.has(fly.id));
    const mourner = followers.length ? rng.pick(followers) : rng.chance(0.5) && living.length ? rng.pick(living) : null;
    if (mourner) post(mourner, writeBuzz(`rip:${victim.cause}`, rng, { handle: victim.handle, age: day(victim.age) }));
  }

  function kill(fly, cause) {
    if (!fly.alive) return;
    fly.alive = false;
    fly.cause = cause;
    fly.diedAt = tickCount;
    stats.deaths++;
    emit({ type: 'death', flyId: fly.id, cause, ageDays: day(fly.age) });
    mourn(fly);
    pruneMemoriam();
  }

  function follow(fly, target) {
    if (fly.id === target.id || fly.following.has(target.id)) return;
    fly.following.add(target.id);
    target.followers.add(fly.id);
    emit({ type: 'follow', flyId: fly.id, targetId: target.id, followers: target.followers.size });
    if (MILESTONES.has(target.followers.size)) post(target, writeBuzz('milestone', rng, { count: target.followers.size }));
  }

  function like(fly, buzz) {
    if (buzz.likedBy.has(fly.id)) return;
    buzz.likedBy.add(fly.id);
    buzz.likes++;
    const author = flies.get(buzz.flyId);
    if (author) author.likes++;
    emit({ type: 'like', flyId: fly.id, buzzId: buzz.id, authorId: buzz.flyId, likes: buzz.likes });
  }

  function flirt(fly, target) {
    post(fly, writeBuzz('flirt', rng, { handle: target.handle }));
    fly.crushId = target.id;
    if (target.crushId === fly.id) {
      post(target, writeBuzz('couple', rng, { handle: fly.handle }));
      eggs.push({ hatchAt: tickCount + EGG_TICKS, parents: [fly, target] });
      fly.crushId = null;
      target.crushId = null;
    } else if (rng.chance(target.traits.romantic)) {
      target.crushId = fly.id;
    }
  }

  function recentBuzzesFor(fly) {
    return buzzes.slice(0, 12).filter((buzz) => buzz.flyId !== fly.id && flies.get(buzz.flyId)?.alive);
  }

  function act(fly) {
    const neighbors = at(fly.spot).filter((other) => other.id !== fly.id);
    const recentBuzzes = recentBuzzesFor(fly);
    const atFood = SPOT_BY_ID.get(fly.spot).kind === 'food';
    const action = chooseAction(fly, { neighbors, recentBuzzes, atFood }, rng);
    if (action === 'move') {
      moveFly(fly, chooseDestination(fly, snacks, rng));
      if (fly.spot === 'web') post(fly, writeBuzz('spot:web', rng));
    } else if (action === 'buzz') {
      post(fly, spotBuzz(fly.spot, rng));
    } else if (action === 'reply') {
      const target = rng.pick(recentBuzzes);
      post(fly, writeBuzz('reply', rng, { handle: target.handle }), target.id);
    } else if (action === 'like') {
      const followed = recentBuzzes.filter((buzz) => fly.following.has(buzz.flyId));
      like(fly, rng.pick(followed.length && rng.chance(0.7) ? followed : recentBuzzes));
    } else if (action === 'follow') {
      const target = rng.pick(neighbors.filter((other) => !fly.following.has(other.id)));
      follow(fly, target);
      if (rng.chance(target.traits.social * 0.5)) follow(target, fly);
    } else if (action === 'flirt') {
      flirt(fly, rng.pick(neighbors));
    }
  }

  function resolveWeb(fly) {
    if (fly.spot !== 'web' || tickCount - fly.arrivedAt !== WEB_RESOLVE_TICKS) return;
    if (rng.chance(WEB_CATCH_CHANCE)) {
      post(fly, writeBuzz('lastWords', rng));
      kill(fly, 'spider');
    } else {
      post(fly, writeBuzz('escapeWeb', rng));
      moveFly(fly, rng.pick(SAFE_SPOTS));
    }
  }

  function live(fly) {
    fly.age++;
    const eating = SPOT_BY_ID.get(fly.spot).kind === 'food';
    fly.hunger = Math.min(1, Math.max(0, fly.hunger + fly.hungerRate - (eating ? 0.12 : 0)));
    if (isDueToDie(fly)) kill(fly, 'age');
    else resolveWeb(fly);
  }

  function hatchEggs() {
    for (let i = eggs.length - 1; i >= 0; i--) {
      if (eggs[i].hatchAt > tickCount) continue;
      const [clutch] = eggs.splice(i, 1);
      const size = 1 + Math.floor(rng.next() * MAX_CLUTCH);
      for (let n = 0; n < size && alive().length < config.maxFlies; n++) hatch('trash', clutch.parents);
    }
    if (alive().length < config.minFlies) hatch('trash');
  }

  function statsView() {
    return { ...stats, alive: alive().length, tick: tickCount, day: day(tickCount) };
  }

  function memoriam() {
    return [...flies.values()]
      .filter((fly) => !fly.alive)
      .sort((a, b) => b.diedAt - a.diedAt)
      .map(publicFly);
  }

  function tick() {
    tickCount++;
    for (const fly of alive()) live(fly);
    for (const spotId of Object.keys(snacks)) {
      snacks[spotId]--;
      if (snacks[spotId] <= 0) delete snacks[spotId];
    }
    hatchEggs();
    for (const fly of rng.shuffle(alive())) {
      if (fly.alive && rng.chance(config.actChance)) act(fly);
    }
    emit({
      type: 'tick',
      stats: statsView(),
      trending: trendingHashtags(buzzes.slice(0, 80)),
      top: rankFlies(alive()).map(publicFly),
      memoriam: memoriam(),
    });
  }

  function requireSpot(spotId) {
    const spot = SPOT_BY_ID.get(spotId);
    if (!spot) throw new SimulationError(`unknown spot ${spotId}`);
    return spot;
  }

  function swat(spotId) {
    const spot = requireSpot(spotId);
    emit({ type: 'swat', spot: spot.id });
    const targets = at(spot.id);
    const bystanders = alive().filter((fly) => fly.spot !== spot.id);
    if (!targets.length) {
      if (bystanders.length) post(rng.pick(bystanders), writeBuzz('swatMiss', rng, { spot: spot.label }));
      return { spot: spot.id, killed: 0, survived: 0 };
    }
    const { killed, survivors } = resolveSwat(targets, rng);
    for (const fly of killed) kill(fly, 'swatter');
    for (const fly of survivors) {
      moveFly(fly, rng.pick(SAFE_SPOTS.filter((id) => id !== spot.id)));
    }
    if (survivors.length) post(rng.pick(survivors), writeBuzz('swatDodge', rng, { spot: spot.label }));
    if (bystanders.length) post(rng.pick(bystanders), writeBuzz('swatAlert', rng, { spot: spot.label.toUpperCase() }));
    return { spot: spot.id, killed: killed.length, survived: survivors.length };
  }

  function dropSnack(spotId) {
    const spot = requireSpot(spotId);
    if (spot.kind !== 'food') throw new SimulationError(`snacks only land on food spots, ${spot.id} is ${spot.kind}`);
    const snack = rng.pick(SNACKS);
    snacks[spot.id] = SNACK_TICKS;
    emit({ type: 'snack', spot: spot.id, snack, ticks: SNACK_TICKS });
    const living = alive();
    if (living.length) post(rng.pick(living), writeBuzz('snack', rng, { snack: snack.toUpperCase(), spot: spot.label.toUpperCase() }));
    return { spot: spot.id, snack, ticks: SNACK_TICKS };
  }

  function snapshot() {
    return {
      spots: SPOTS,
      species: SPECIES,
      flies: alive().map(publicFly),
      feed: buzzes.slice(0, 60).map(publicBuzz),
      stats: statsView(),
      trending: trendingHashtags(buzzes.slice(0, 80)),
      top: rankFlies(alive()).map(publicFly),
      memoriam: memoriam(),
    };
  }

  function profile(flyId) {
    const fly = flies.get(flyId);
    if (!fly) return null;
    return {
      ...publicFly(fly),
      followingHandles: [...fly.following].map((id) => flies.get(id)?.handle).filter(Boolean),
      buzzes: buzzes.filter((buzz) => buzz.flyId === fly.id).slice(0, 8).map(publicBuzz),
    };
  }

  for (let i = 0; i < config.startFlies; i++) {
    const fly = createFly(rng.pick(SAFE_SPOTS));
    fly.age = Math.floor(fly.lifespan * rng.next() * 0.6);
  }

  return { tick, swat, dropSnack, snapshot, profile, subscribe };
}
