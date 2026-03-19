import { defineConfig } from "vite";
import { remix } from "@remix-run/dev";
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
