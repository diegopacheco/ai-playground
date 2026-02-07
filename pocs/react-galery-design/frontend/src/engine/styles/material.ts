import type { StyleConfig } from "../../types";

export const materialStyle: StyleConfig = {
  name: "Material",
  cssVariables: {
    "--primary": "#6200ea",
    "--secondary": "#03dac6",
    "--bg": "#ffffff",
    "--text": "#212121",
    "--font": "'Roboto', -apple-system, BlinkMacSystemFont, sans-serif",
    "--radius": "4px",
    "--shadow": "0 2px 4px rgba(0,0,0,0.14), 0 3px 4px rgba(0,0,0,0.12)",
    "--spacing": "24px",
    "--accent": "#03dac6",
  },
  bodyStyles: `
body { background: #fafafa; }
button { background: #6200ea; color: #fff; border-radius: 4px; padding: 12px 24px; text-transform: uppercase; font-weight: 500; letter-spacing: 1px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
button:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
  `,
  componentOverrides: `
.card { box-shadow: 0 2px 4px rgba(0,0,0,0.14), 0 3px 4px rgba(0,0,0,0.12); }
nav { box-shadow: 0 2px 4px rgba(0,0,0,0.14); }
  `,
};
