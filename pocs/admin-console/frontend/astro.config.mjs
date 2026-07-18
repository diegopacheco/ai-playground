import { defineConfig } from "astro/config";
import react from "@astrojs/react";

export default defineConfig({
  integrations: [react()],
  devToolbar: { enabled: false },
  server: { host: "0.0.0.0", port: 4321 },
  vite: {
    server: {
      proxy: {
        "/api": { target: "http://localhost:8099", changeOrigin: true },
        "/swagger": { target: "http://localhost:8099", changeOrigin: true },
        "/swagger-ui": { target: "http://localhost:8099", changeOrigin: true },
        "/v3": { target: "http://localhost:8099", changeOrigin: true }
      }
    }
  }
});
