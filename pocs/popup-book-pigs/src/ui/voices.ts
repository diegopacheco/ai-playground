export interface VoiceInfo {
  name: string;
  lang: string;
}

const NATURAL_MARKERS = ["natural", "neural", "premium", "enhanced", "online"];

const PREFERRED_NAMES = [
  "Google US English",
  "Ava",
  "Samantha",
  "Allison",
  "Susan",
  "Zoe",
  "Karen",
  "Moira",
  "Daniel",
  "Google UK English Female",
];

const NOVELTY_NAMES = [
  "Albert",
  "Bad News",
  "Bahh",
  "Bells",
  "Boing",
  "Bubbles",
  "Cellos",
  "Fred",
  "Good News",
  "Jester",
  "Junior",
  "Organ",
  "Ralph",
  "Superstar",
  "Trinoids",
  "Whisper",
  "Wobble",
  "Zarvox",
  "Grandma",
  "Grandpa",
];

function isNovelty(name: string): boolean {
  return NOVELTY_NAMES.some((n) => name.startsWith(n));
}

function score(voice: VoiceInfo): number {
  const name = voice.name.toLowerCase();
  if (isNovelty(voice.name)) return -1;
  const natural = NATURAL_MARKERS.some((m) => name.includes(m)) ? 100 : 0;
  const rank = PREFERRED_NAMES.findIndex((n) => name.startsWith(n.toLowerCase()));
  const preferred = rank === -1 ? 0 : 50 - rank;
  const american = voice.lang.replace("_", "-") === "en-US" ? 5 : 0;
  return natural + preferred + american;
}

export function pickVoice<T extends VoiceInfo>(voices: T[]): T | undefined {
  const english = voices.filter((v) => v.lang.toLowerCase().startsWith("en") && !isNovelty(v.name));
  return english.reduce<T | undefined>((best, v) => (!best || score(v) > score(best) ? v : best), undefined);
}
