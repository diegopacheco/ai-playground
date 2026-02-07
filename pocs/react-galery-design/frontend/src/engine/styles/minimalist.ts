import type { StyleConfig } from "../../types";

export const minimalistStyle: StyleConfig = {
  name: "Minimalist",
  cssVariables: {
    "--primary": "#111111",
    "--secondary": "#666666",
    "--bg": "#ffffff",
    "--text": "#111111",
    "--font": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, sans-serif",
    "--radius": "2px",
    "--shadow": "none",
    "--spacing": "40px",
    "--accent": "#111111",
  },
  bodyStyles: `
body { background: #fff; }
button { background: #111; color: #fff; border-radius: 2px; padding: 12px 32px; font-weight: 400; letter-spacing: 1px; }
button:hover { background: #333; }
  `,
  componentOverrides: `
.card { border: 1px solid #eee; }
nav { border-bottom: 1px solid #eee; }
h1, h2, h3 { font-weight: 300; }
  `,
};
