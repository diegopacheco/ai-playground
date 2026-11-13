import { pickVoice } from "./voices";

export const canNarrate = typeof window !== "undefined" && "speechSynthesis" in window;

let voice: SpeechSynthesisVoice | undefined;

function loadVoice(): void {
  voice = pickVoice(window.speechSynthesis.getVoices());
}

if (canNarrate) {
  loadVoice();
  window.speechSynthesis.addEventListener("voiceschanged", loadVoice);
}

export function speak(text: string): void {
  if (!canNarrate) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = voice?.lang ?? "en-US";
  utterance.rate = 0.95;
  utterance.pitch = 1;
  if (voice) utterance.voice = voice;
  window.speechSynthesis.speak(utterance);
}

export function hush(): void {
  if (canNarrate) window.speechSynthesis.cancel();
}
