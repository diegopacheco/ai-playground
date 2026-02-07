import type { StyleConfig } from "../../types";

export const terminalStyle: StyleConfig = {
  name: "Terminal",
  cssVariables: {
    "--primary": "#00ff41",
    "--secondary": "#ffb000",
    "--bg": "#0a0a0a",
    "--text": "#00ff41",
    "--font": "'Courier New', Courier, monospace",
    "--radius": "0px",
    "--shadow": "0 0 10px rgba(0,255,65,0.3)",
    "--spacing": "24px",
    "--accent": "#ffb000",
  },
  bodyStyles: `
body { background: #0a0a0a; color: #00ff41; }
a { color: #ffb000; }
button { background: transparent; border: 1px solid #00ff41; color: #00ff41; }
button:hover { background: #00ff41; color: #0a0a0a; }
input, textarea, select { background: #111; border: 1px solid #00ff41; color: #00ff41; }
  `,
  componentOverrides: `
nav { border-bottom: 1px solid #00ff41; }
.card { border: 1px solid #00ff41; }
  `,
};
