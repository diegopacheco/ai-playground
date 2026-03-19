import { defineConfig } from "vite";
import { remixVitePlugin as remix } from "@remix-run/dev/dist/vite/plugin.js";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [remix(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
    },
  },
});
