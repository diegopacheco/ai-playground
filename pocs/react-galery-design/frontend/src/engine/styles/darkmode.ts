import type { StyleConfig } from "../../types";

export const darkmodeStyle: StyleConfig = {
  name: "Dark Mode",
  cssVariables: {
    "--primary": "#14b8a6",
    "--secondary": "#6366f1",
    "--bg": "#121212",
    "--text": "#e0e0e0",
    "--font": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "--radius": "8px",
    "--shadow": "0 2px 8px rgba(0,0,0,0.4)",
    "--spacing": "24px",
    "--accent": "#6366f1",
  },
  bodyStyles: `
body { background: #121212; color: #e0e0e0; }
button { background: #14b8a6; color: #121212; border-radius: 8px; padding: 12px 24px; font-weight: 600; }
button:hover { background: #0d9488; }
a { color: #14b8a6; }
input, textarea, select { background: #1e1e1e; border: 1px solid #333; color: #e0e0e0; }
  `,
  componentOverrides: `
.card { background: #1e1e1e; border: 1px solid #2a2a2a; }
nav { background: #1a1a1a; border-bottom: 1px solid #2a2a2a; }
  `,
};
