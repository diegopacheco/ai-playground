import type { StyleConfig } from "../../types";

export const playfulStyle: StyleConfig = {
  name: "Playful",
  cssVariables: {
    "--primary": "#ff6b35",
    "--secondary": "#7c3aed",
    "--bg": "#fff9e6",
    "--text": "#2d1b69",
    "--font": "'Nunito', 'Comic Sans MS', cursive, sans-serif",
    "--radius": "20px",
    "--shadow": "0 4px 14px rgba(255,107,53,0.25)",
    "--spacing": "28px",
    "--accent": "#7c3aed",
  },
  bodyStyles: `
body { background: #fff9e6; }
button { background: #ff6b35; color: #fff; border-radius: 999px; padding: 12px 28px; font-weight: 700; }
button:hover { background: #e85d2c; transform: scale(1.05); transition: transform 150ms; }
  `,
  componentOverrides: `
.card { border-radius: 20px; border: 3px solid #7c3aed; }
nav { border-bottom: 3px dashed #ff6b35; }
h1, h2, h3 { color: #7c3aed; }
  `,
};
