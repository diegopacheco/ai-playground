import type { StyleConfig } from "../../types";

export const neomorphismStyle: StyleConfig = {
  name: "Neomorphism",
  cssVariables: {
    "--primary": "#6c8ebf",
    "--secondary": "#a0b4c8",
    "--bg": "#e0e5ec",
    "--text": "#44475a",
    "--font": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    "--radius": "16px",
    "--shadow": "8px 8px 16px #b8bec7, -8px -8px 16px #ffffff",
    "--spacing": "28px",
    "--accent": "#6c8ebf",
  },
  bodyStyles: `
body { background: #e0e5ec; }
button { background: #e0e5ec; color: #44475a; box-shadow: 4px 4px 8px #b8bec7, -4px -4px 8px #ffffff; border-radius: 16px; padding: 12px 24px; font-weight: 600; }
button:hover { box-shadow: 2px 2px 4px #b8bec7, -2px -2px 4px #ffffff; }
input, textarea, select { background: #e0e5ec; border: none; box-shadow: inset 4px 4px 8px #b8bec7, inset -4px -4px 8px #ffffff; border-radius: 12px; padding: 12px 16px; }
  `,
  componentOverrides: `
.card { box-shadow: 8px 8px 16px #b8bec7, -8px -8px 16px #ffffff; border: none; }
  `,
};
