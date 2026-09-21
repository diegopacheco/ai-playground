import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const uiPort = Number(process.env.UI_PORT ?? 5174);
const apiPort = Number(process.env.API_PORT ?? 8787);

export default defineConfig({
  plugins: [react()],
  server: {
    port: uiPort,
    strictPort: true,
    proxy: { "/api": `http://127.0.0.1:${apiPort}` },
  },
});
