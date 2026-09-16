import type { Pilot, Side } from "@fly-zord/engine";

export interface PilotChoice {
  readonly id: string;
  readonly label: string;
  readonly kind: "human" | "fly";
  readonly model: string;
}

export const PILOT_CHOICES: readonly PilotChoice[] = [
  { id: "human", label: "Human at the sticks", kind: "human", model: "human" },
  { id: "instinct", label: "Fly on instinct", kind: "fly", model: "instinct" },
  { id: "claude", label: "Fly wired to Claude Code", kind: "fly", model: "claude-opus-5" },
  { id: "codex", label: "Fly wired to Codex", kind: "fly", model: "gpt-6-astra" },
  { id: "agy", label: "Fly wired to Agy", kind: "fly", model: "gemini-3.8-flash-low" },
  { id: "ollama", label: "Fly wired to Ollama", kind: "fly", model: "llama3.2" }
];

export const FLY_NAMES: Readonly<Record<Side, string>> = { left: "Buzzbolt", right: "Wingnut" };

export function makePilot(side: Side, choiceId: string, model: string): Pilot {
  const choice = PILOT_CHOICES.find(entry => entry.id === choiceId) ?? PILOT_CHOICES[0]!;
  return { name: choice.kind === "human" ? "Ranger" : FLY_NAMES[side], kind: choice.kind, provider: choice.id, model };
}
