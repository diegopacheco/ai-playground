import type { StyleConfig } from "../../types";

export const traditionalStyle: StyleConfig = {
  name: "Traditional",
  cssVariables: {
    "--primary": "#3d2b1f",
    "--secondary": "#8b6914",
    "--bg": "#faf8f5",
    "--text": "#2c1810",
    "--font": "Georgia, 'Times New Roman', Times, serif",
    "--radius": "4px",
    "--shadow": "0 1px 3px rgba(0,0,0,0.12)",
    "--spacing": "28px",
    "--accent": "#8b6914",
  },
  bodyStyles: `
body { background: #faf8f5; }
button { background: #3d2b1f; color: #faf8f5; border-radius: 4px; padding: 10px 20px; }
button:hover { background: #5a4030; }
  `,
  componentOverrides: `
nav { border-bottom: 2px solid #3d2b1f; }
h1, h2, h3 { font-family: Georgia, serif; }
  `,
};
