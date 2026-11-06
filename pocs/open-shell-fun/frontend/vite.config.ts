import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = process.env.BACKEND_PORT ?? "8091";

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.FRONTEND_PORT ?? 5191),
    strictPort: true,
    proxy: { "/api": { target: `http://localhost:${backend}`, ws: true } },
  },
});
