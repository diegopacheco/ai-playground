export const colors = {
  bg: "#FBF6EF",
  surface: "#F4EADD",
  surfaceRaised: "#FFFFFF",
  border: "#E6D5C1",
  borderStrong: "#D4BCA1",
  text: "#2B1D16",
  muted: "#8B7466",
  accent: "#B2542F",
  accentStrong: "#8F3F20",
  accentWash: "#F7E6DA",
  accentSoft: "#E0A73F",
  ok: "#5F8A4A",
  error: "#C0392B"
} as const;

export type ColorToken = keyof typeof colors;

export const cssVariables = Object.entries(colors)
  .map(([name, value]) => `--${name.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase())}: ${value};`)
  .join("\n  ");
