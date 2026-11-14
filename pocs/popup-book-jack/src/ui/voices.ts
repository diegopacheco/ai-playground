export interface VoiceInfo {
  name: string;
  lang: string;
}

const NATURAL_MARKERS = ["natural", "neural", "premium", "enhanced", "online", "network"];

const PREFERRED_NAMES = ["Ava", "Jenny", "Aria", "Google US English", "Google UK English Female", "Samantha", "Daniel", "Serena", "Karen", "Moira"];

const ROBOTIC_NAMES = [
  "Eddy", "Flo", "Grandma", "Grandpa", "Reed", "Rocko", "Sandy", "Shelley",
  "Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Fred", "Good News", "Jester", "Junior", "Kathy", "Organ", "Ralph", "Superstar", "Trinoids", "Whisper", "Wobble", "Zarvox",
];

function isRobotic(name: string): boolean {
  return ROBOTIC_NAMES.some((n) => name === n || name.startsWith(`${n} `));
}

function normalLang(voice: VoiceInfo): string {
  return voice.lang.replace("_", "-").toLowerCase();
}

function score(voice: VoiceInfo): number {
  const name = voice.name.toLowerCase();
  const natural = NATURAL_MARKERS.some((m) => name.includes(m)) ? 100 : 0;
  const rank = PREFERRED_NAMES.findIndex((n) => name.includes(n.toLowerCase()));
  const preferred = rank === -1 ? 0 : 50 - rank;
  const lang = normalLang(voice);
  const native = lang === "en-us" || lang === "en-gb" ? 60 : 0;
  return natural + preferred + native;
}

export function pickVoice<T extends VoiceInfo>(voices: T[]): T | undefined {
  const english = voices.filter((v) => normalLang(v).startsWith("en") && !isRobotic(v.name));
  return english.reduce<T | undefined>((best, v) => (!best || score(v) > score(best) ? v : best), undefined);
}
