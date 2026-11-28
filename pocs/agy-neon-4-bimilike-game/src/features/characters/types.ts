export const CHARACTER_IDS = ["momo", "pip", "bubu", "lulu", "kiki", "nori"] as const;
export type CharacterId = (typeof CHARACTER_IDS)[number];

export type CharacterSpecies = "bear" | "bunny" | "frog" | "fox" | "owl" | "kitty";

export type ActivityStats = {
  readonly cookedCount: number;
  readonly paintedCount: number;
  readonly movedCount: number;
  readonly calledCount: number;
  readonly fedCount: number;
};

export type UnlockRule =
  | { readonly kind: "starter" }
  | { readonly kind: "cook"; readonly target: number }
  | { readonly kind: "paint"; readonly target: number }
  | { readonly kind: "move"; readonly target: number }
  | { readonly kind: "call"; readonly target: number };

export type CharacterSpec = {
  readonly id: CharacterId;
  readonly name: string;
  readonly species: CharacterSpecies;
  readonly emoji: string;
  readonly bodyColor: string;
  readonly bellyColor: string;
  readonly accentColor: string;
  readonly voicePitch: number;
  readonly voiceRate: number;
  readonly chirpMultiplier: number;
  readonly favoriteFood: string;
  readonly greeting: string;
  readonly unlockHint: string;
  readonly unlockRule: UnlockRule;
  readonly homeX: number;
  readonly homeZ: number;
};

export type CharacterPlacement = {
  readonly id: CharacterId;
  readonly x: number;
  readonly z: number;
  readonly mood: "happy" | "yummy" | "dancing" | "waving";
  readonly moodUntil: number;
};
