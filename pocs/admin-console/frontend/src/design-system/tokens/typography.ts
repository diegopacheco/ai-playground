export const typography = {
  sans: "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
  mono: "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace",
  sizeXs: "11px",
  sizeSm: "12px",
  sizeMd: "13px",
  sizeLg: "15px",
  sizeXl: "19px",
  weightRegular: "400",
  weightMedium: "500",
  weightBold: "600"
} as const;

export type TypographyToken = keyof typeof typography;
