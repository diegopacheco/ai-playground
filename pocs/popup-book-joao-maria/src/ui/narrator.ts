import { pickVoice } from "./voices";

export const canNarrate = typeof window !== "undefined" && "speechSynthesis" in window;

const RATE = 0.95;
const PITCH = 1;

let voice: SpeechSynthesisVoice | undefined;

function loadVoice(): void {
  voice = pickVoice(window.speechSynthesis.getVoices());
}

function currentVoice(): SpeechSynthesisVoice | undefined {
  if (!voice) loadVoice();
  return voice;
}

if (canNarrate) {
  loadVoice();
  window.speechSynthesis.onvoiceschanged = loadVoice;
}

export function voiceName(): string {
  return canNarrate ? (currentVoice()?.name ?? "pt-BR") : "";
}

export function unlock(): void {
  if (!canNarrate) return;
  const silent = new SpeechSynthesisUtterance(" ");
  silent.volume = 0;
  silent.lang = "pt-BR";
  window.speechSynthesis.speak(silent);
}

export function speak(parts: string[]): void {
  if (!canNarrate) return;
  const synth = window.speechSynthesis;
  synth.cancel();
  synth.resume();
  const chosen = currentVoice();
  for (const part of parts) {
    const utterance = new SpeechSynthesisUtterance(part);
    utterance.lang = chosen?.lang.replace("_", "-") ?? "pt-BR";
    if (chosen) utterance.voice = chosen;
    utterance.rate = RATE;
    utterance.pitch = PITCH;
    synth.speak(utterance);
  }
}

export function hush(): void {
  if (canNarrate) window.speechSynthesis.cancel();
}
