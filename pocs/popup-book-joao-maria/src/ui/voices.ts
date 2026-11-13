export interface VoiceInfo {
  name: string;
  lang: string;
}

const NATURAL_MARKERS = ["natural", "neural", "premium", "enhanced", "online", "network"];

const PREFERRED_NAMES = ["Francisca", "Thalita", "Antonio", "Google português do Brasil", "Luciana", "Fernanda", "Felipe", "Joana", "Catarina"];

const ROBOTIC_NAMES = [
  "Eddy", "Flo", "Grandma", "Grandpa", "Reed", "Rocko", "Sandy", "Shelley",
  "Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Good News", "Jester", "Organ", "Superstar", "Trinoids", "Whisper", "Wobble", "Zarvox",
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
  const brazilian = normalLang(voice) === "pt-br" ? 60 : 0;
  return natural + preferred + brazilian;
}

export function pickVoice<T extends VoiceInfo>(voices: T[]): T | undefined {
  const portuguese = voices.filter((v) => normalLang(v).startsWith("pt") && !isRobotic(v.name));
  return portuguese.reduce<T | undefined>((best, v) => (!best || score(v) > score(best) ? v : best), undefined);
}
