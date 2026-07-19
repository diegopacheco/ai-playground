export const spacing = {
  none: "0",
  xs: "4px",
  sm: "8px",
  md: "12px",
  lg: "16px",
  xl: "24px",
  xxl: "32px"
} as const;

export type SpacingToken = keyof typeof spacing;

export const radius = {
  sm: "4px",
  md: "6px",
  lg: "10px"
} as const;
