import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { tanstackRouter } from "@tanstack/router-plugin/vite";

function port(name: string): number {
  const value = Number(process.env[name]);
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${name} must be set to a port`);
  return value;
}

export default defineConfig(({ command }) => {
  const ports = command === "serve" ? { ui: port("FRONTEND_PORT"), api: port("BACKEND_PORT") } : undefined;
  return {
    plugins: [tanstackRouter({ target: "react", autoCodeSplitting: true }), react()],
    server: ports && {
      port: ports.ui,
      strictPort: true,
      proxy: { "/api": `http://localhost:${ports.api}` },
    },
  };
});
