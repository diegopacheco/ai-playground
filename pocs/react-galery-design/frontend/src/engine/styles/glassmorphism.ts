import type { StyleConfig } from "../../types";

export const glassmorphismStyle: StyleConfig = {
  name: "Glassmorphism",
  cssVariables: {
    "--primary": "#6366f1",
    "--secondary": "#a855f7",
    "--bg": "rgba(255,255,255,0.15)",
    "--text": "#ffffff",
    "--font": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    "--radius": "16px",
    "--shadow": "0 8px 32px rgba(31,38,135,0.37)",
    "--spacing": "28px",
    "--accent": "#a855f7",
  },
  bodyStyles: `
body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: #fff; }
button { background: rgba(255,255,255,0.2); color: #fff; backdrop-filter: blur(4px); border: 1px solid rgba(255,255,255,0.3); border-radius: 16px; padding: 12px 24px; }
button:hover { background: rgba(255,255,255,0.3); }
input, textarea, select { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); color: #fff; backdrop-filter: blur(4px); }
  `,
  componentOverrides: `
.card, nav, section { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2); border-radius: 16px; }
a { color: #e0e7ff; }
  `,
};
