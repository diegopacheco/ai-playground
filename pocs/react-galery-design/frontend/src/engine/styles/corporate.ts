import type { StyleConfig } from "../../types";

export const corporateStyle: StyleConfig = {
  name: "Corporate",
  cssVariables: {
    "--primary": "#1e3a5f",
    "--secondary": "#2563eb",
    "--bg": "#ffffff",
    "--text": "#1f2937",
    "--font": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    "--radius": "6px",
    "--shadow": "0 1px 3px rgba(0,0,0,0.1)",
    "--spacing": "24px",
    "--accent": "#2563eb",
  },
  bodyStyles: `
body { background: #f3f4f6; }
button { background: #1e3a5f; color: #fff; border-radius: 6px; padding: 10px 24px; }
button:hover { background: #2a4a73; }
  `,
  componentOverrides: `
nav { background: #1e3a5f; }
nav a { color: #fff; }
nav .brand { color: #fff; }
  `,
};
