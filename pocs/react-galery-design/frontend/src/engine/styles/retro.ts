import type { StyleConfig } from "../../types";

export const retroStyle: StyleConfig = {
  name: "Retro",
  cssVariables: {
    "--primary": "#ff00ff",
    "--secondary": "#00ffff",
    "--bg": "#1a0533",
    "--text": "#ffffff",
    "--font": "'Courier New', Courier, monospace",
    "--radius": "0px",
    "--shadow": "0 0 20px rgba(255,0,255,0.5)",
    "--spacing": "24px",
    "--accent": "#00ffff",
  },
  bodyStyles: `
body { background: #1a0533; color: #fff; }
button { background: transparent; border: 2px solid #ff00ff; color: #ff00ff; padding: 12px 24px; text-transform: uppercase; letter-spacing: 2px; }
button:hover { background: #ff00ff; color: #1a0533; box-shadow: 0 0 20px #ff00ff; }
a { color: #00ffff; }
input, textarea, select { background: #2a1543; border: 2px solid #ff00ff; color: #fff; }
  `,
  componentOverrides: `
.card { border: 2px solid #ff00ff; box-shadow: 0 0 15px rgba(255,0,255,0.3); }
nav { border-bottom: 2px solid #ff00ff; }
h1, h2, h3 { text-shadow: 0 0 10px #ff00ff; }
  `,
};
