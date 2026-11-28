export type EarShape = "round" | "long" | "pointy" | "none";

export type Extra = "none" | "beak" | "tentacles" | "spikes" | "horn" | "tail" | "fin";

export type PlushLook = {
  readonly name: string;
  readonly emoji: string;
  readonly fur: number;
  readonly trim: number;
  readonly accent: number;
  readonly ears: EarShape;
  readonly extra: Extra;
};

export const PLUSH_LOOKS: readonly PlushLook[] = [
  { name: "bear", emoji: "🐻", fur: 0x9a6234, trim: 0x8a552b, accent: 0xe8c9a0, ears: "round", extra: "none" },
  { name: "bunny", emoji: "🐰", fur: 0xf4f0f5, trim: 0xf4f0f5, accent: 0xffa8c8, ears: "long", extra: "none" },
  { name: "cat", emoji: "🐱", fur: 0xf29a3f, trim: 0xe0862b, accent: 0xfff1dc, ears: "pointy", extra: "none" },
  { name: "panda", emoji: "🐼", fur: 0xf7f7f7, trim: 0x1d1d22, accent: 0xffffff, ears: "round", extra: "none" },
  { name: "frog", emoji: "🐸", fur: 0x58c45a, trim: 0x45a947, accent: 0xd9f7a8, ears: "none", extra: "none" },
  { name: "chick", emoji: "🐥", fur: 0xffd84a, trim: 0xffc21f, accent: 0xff9a2e, ears: "none", extra: "none" },
  { name: "koala", emoji: "🐨", fur: 0x9aa3b5, trim: 0x838da1, accent: 0xe9edf5, ears: "round", extra: "none" },
  { name: "pig", emoji: "🐷", fur: 0xffa3bd, trim: 0xff8fae, accent: 0xff7fa3, ears: "pointy", extra: "none" },
  { name: "penguin", emoji: "🐧", fur: 0x2f3858, trim: 0x262d48, accent: 0xffffff, ears: "none", extra: "beak" },
  { name: "octopus", emoji: "🐙", fur: 0xb985ff, trim: 0xa36cf0, accent: 0xf1dcff, ears: "none", extra: "tentacles" },
  { name: "dino", emoji: "🦕", fur: 0x5cc8b8, trim: 0x43ab9c, accent: 0xd9fff3, ears: "none", extra: "spikes" },
  { name: "unicorn", emoji: "🦄", fur: 0xfff7fd, trim: 0xd9c4ff, accent: 0xffc4e1, ears: "pointy", extra: "horn" },
  { name: "fox", emoji: "🦊", fur: 0xe0582a, trim: 0xc94a20, accent: 0xfff4e8, ears: "pointy", extra: "tail" },
  { name: "whale", emoji: "🐳", fur: 0x6fb4ff, trim: 0x5a9ced, accent: 0xe2f2ff, ears: "none", extra: "fin" },
];
