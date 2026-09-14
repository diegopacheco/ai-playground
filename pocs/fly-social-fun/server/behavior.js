import { SPOTS } from './world.js';

export const SWAT_KILL_CHANCE = 0.65;
export const WEB_CATCH_CHANCE = 0.5;
export const SNACK_PULL = 4;

export function destinationWeights(fly, snacks) {
  return SPOTS.filter((spot) => spot.id !== fly.spot).map((spot) => {
    let weight = spot.appeal;
    if (spot.kind === 'food') weight *= 1 + fly.hunger * 6;
    else weight *= 1.2 - fly.hunger;
    if (spot.kind === 'hazard') weight *= fly.traits.reckless * (1 - fly.hunger);
    if (snacks[spot.id]) weight *= SNACK_PULL;
    return { value: spot.id, weight: Math.max(weight, 0) };
  });
}

export function chooseDestination(fly, snacks, rng) {
  return rng.weighted(destinationWeights(fly, snacks));
}

export function actionWeights(fly, { neighbors, recentBuzzes, atFood }) {
  const { chatty, social, romantic } = fly.traits;
  const strangers = neighbors.filter((other) => !fly.following.has(other.id));
  const hasFeed = recentBuzzes.length > 0;
  return [
    { value: 'move', weight: 0.6 + fly.hunger * (atFood ? 0.2 : 3) + (1 - social) * 0.4 },
    { value: 'buzz', weight: 0.4 + chatty * 1.6 },
    { value: 'reply', weight: hasFeed ? chatty * social * 1.5 : 0 },
    { value: 'like', weight: hasFeed ? 0.3 + social * 1.5 : 0 },
    { value: 'follow', weight: strangers.length ? 0.3 + social * 2 : 0 },
    { value: 'flirt', weight: neighbors.length ? romantic * 0.6 : 0 },
  ];
}

export function chooseAction(fly, context, rng) {
  return rng.weighted(actionWeights(fly, context));
}

export function resolveSwat(fliesAtSpot, rng) {
  const killed = [];
  const survivors = [];
  for (const fly of fliesAtSpot) {
    if (rng.chance(SWAT_KILL_CHANCE)) killed.push(fly);
    else survivors.push(fly);
  }
  return { killed, survivors };
}

export function isDueToDie(fly) {
  return fly.age >= fly.lifespan;
}
