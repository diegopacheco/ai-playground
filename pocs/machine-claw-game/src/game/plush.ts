export type EarShape = "round" | "long" | "pointy" | "none";

export type PlushLook = {
  readonly name: string;
  readonly emoji: string;
  readonly fur: number;
  readonly trim: number;
  readonly accent: number;
  readonly ears: EarShape;
};

export const PLUSH_LOOKS: readonly PlushLook[] = [
  { name: "bear", emoji: "🐻", fur: 0x9a6234, trim: 0x8a552b, accent: 0xe8c9a0, ears: "round" },
  { name: "bunny", emoji: "🐰", fur: 0xf4f0f5, trim: 0xf4f0f5, accent: 0xffa8c8, ears: "long" },
  { name: "cat", emoji: "🐱", fur: 0xf29a3f, trim: 0xe0862b, accent: 0xfff1dc, ears: "pointy" },
  { name: "panda", emoji: "🐼", fur: 0xf7f7f7, trim: 0x1d1d22, accent: 0xffffff, ears: "round" },
  { name: "frog", emoji: "🐸", fur: 0x58c45a, trim: 0x45a947, accent: 0xd9f7a8, ears: "none" },
  { name: "chick", emoji: "🐥", fur: 0xffd84a, trim: 0xffc21f, accent: 0xff9a2e, ears: "none" },
  { name: "koala", emoji: "🐨", fur: 0x9aa3b5, trim: 0x838da1, accent: 0xe9edf5, ears: "round" },
  { name: "pig", emoji: "🐷", fur: 0xffa3bd, trim: 0xff8fae, accent: 0xff7fa3, ears: "pointy" },
];
