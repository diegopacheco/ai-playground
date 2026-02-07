import type { StyleConfig } from "../../types";

export const flatStyle: StyleConfig = {
  name: "Flat",
  cssVariables: {
    "--primary": "#27ae60",
    "--secondary": "#2980b9",
    "--bg": "#ecf0f1",
    "--text": "#2c3e50",
    "--font": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "--radius": "3px",
    "--shadow": "none",
    "--spacing": "24px",
    "--accent": "#2980b9",
  },
  bodyStyles: `
body { background: #ecf0f1; }
button { background: #27ae60; color: #fff; border-radius: 3px; padding: 12px 24px; font-weight: 600; }
button:hover { background: #219a52; }
  `,
  componentOverrides: `
.card { background: #fff; border: none; box-shadow: none; }
nav { background: #2c3e50; }
nav a, nav .brand { color: #ecf0f1; }
  `,
};
