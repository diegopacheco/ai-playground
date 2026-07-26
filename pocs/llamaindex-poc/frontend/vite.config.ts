import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5199,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8077",
        changeOrigin: true,
      },
    },
  },
});
