import type { StyleConfig } from "../../types";

export const modernStyle: StyleConfig = {
  name: "Modern",
  cssVariables: {
    "--primary": "#3b82f6",
    "--secondary": "#8b5cf6",
    "--bg": "#ffffff",
    "--text": "#1f2937",
    "--font": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "--radius": "12px",
    "--shadow": "0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06)",
    "--spacing": "32px",
    "--accent": "#8b5cf6",
  },
  bodyStyles: `
body { background: #f9fafb; }
button { background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; border-radius: 12px; padding: 12px 24px; }
button:hover { opacity: 0.9; }
  `,
  componentOverrides: "",
};
