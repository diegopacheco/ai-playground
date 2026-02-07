import type { StyleConfig } from "../../types";

export const brutalistStyle: StyleConfig = {
  name: "Brutalist",
  cssVariables: {
    "--primary": "#000000",
    "--secondary": "#ff0000",
    "--bg": "#ffffff",
    "--text": "#000000",
    "--font": "'Courier New', Courier, monospace",
    "--radius": "0px",
    "--shadow": "none",
    "--spacing": "24px",
    "--accent": "#ff0000",
  },
  bodyStyles: `
body { background: #ffffff; }
button { background: #000; color: #fff; border: 3px solid #000; padding: 12px 24px; text-transform: uppercase; font-weight: 900; }
button:hover { background: #fff; color: #000; }
  `,
  componentOverrides: `
.card { border: 3px solid #000; }
nav { border-bottom: 3px solid #000; }
section { border-bottom: 3px solid #000; }
h1, h2, h3 { text-transform: uppercase; font-weight: 900; letter-spacing: 2px; }
  `,
};
