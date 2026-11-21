import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { tanstackRouter } from "@tanstack/router-plugin/vite";

function requiredPort(name: string): number {
  const value = Number(process.env[name]);
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${name} is required`);
  return value;
}

export default defineConfig(() => {
  const apiPort = requiredPort("API_PORT");
  const uiPort = requiredPort("UI_PORT");
  return {
    plugins: [tanstackRouter({ target: "react", autoCodeSplitting: false }), react()],
    server: {
      port: uiPort,
      strictPort: true,
      host: "127.0.0.1",
      proxy: { "/api": `http://127.0.0.1:${apiPort}` },
    },
  };
});
