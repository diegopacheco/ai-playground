import { pickVoice } from "./voices";

export const canNarrate = typeof window !== "undefined" && "speechSynthesis" in window;

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

export function unlock(): void {
  if (!canNarrate) return;
  const silent = new SpeechSynthesisUtterance(" ");
  silent.volume = 0;
  silent.lang = "en-US";
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
    utterance.lang = chosen?.lang.replace("_", "-") ?? "en-US";
    utterance.rate = 0.95;
    utterance.pitch = 1;
    if (chosen) utterance.voice = chosen;
    synth.speak(utterance);
  }
}

export function hush(): void {
  if (canNarrate) window.speechSynthesis.cancel();
}
