export const LAPS = 3;

export function createRace(track, ids, laps = LAPS) {
  return {
    laps,
    count: track.count,
    time: 0,
    entries: ids.map((id) => ({ id, lap: 0, i: 0, halfway: false, finished: false, finishTime: 0, lapStart: 0, lapTimes: [] })),
  };
}

export function updateEntry(race, entry, i) {
  const n = race.count;
  const prev = entry.i;
  entry.i = i;
  if (entry.finished) return;
  if (i > n * 0.45 && i < n * 0.55) entry.halfway = true;
  if (prev > n * 0.75 && i < n * 0.25 && entry.halfway) {
    entry.lap++;
    entry.halfway = false;
    entry.lapTimes.push(race.time - entry.lapStart);
    entry.lapStart = race.time;
    if (entry.lap >= race.laps) {
      entry.finished = true;
      entry.finishTime = race.time;
    }
  }
}

export function progressOf(race, entry) {
  const base = entry.halfway || entry.i < race.count * 0.5 ? entry.i : entry.i - race.count;
  return entry.lap * race.count + base;
}

export function standings(race) {
  return [...race.entries].sort((a, b) => {
    if (a.finished && b.finished) return a.finishTime - b.finishTime;
    if (a.finished !== b.finished) return a.finished ? -1 : 1;
    return progressOf(race, b) - progressOf(race, a);
  });
}

export function positionOf(race, id) {
  return standings(race).findIndex((e) => e.id === id) + 1;
}

export function formatTime(t) {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, '0')}`;
}
