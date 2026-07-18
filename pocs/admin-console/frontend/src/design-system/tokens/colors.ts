export const colors = {
  bg: "#FAF7F2",
  surface: "#F3EDE4",
  surfaceRaised: "#FBF8F3",
  border: "#DFD3C3",
  text: "#3E322A",
  muted: "#7A6A5D",
  accent: "#8B5E3C",
  accentSoft: "#C9A227",
  ok: "#6B8E4E",
  error: "#A94F3C"
} as const;

export type ColorToken = keyof typeof colors;

export const cssVariables = Object.entries(colors)
  .map(([name, value]) => `--${name.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase())}: ${value};`)
  .join("\n  ");
