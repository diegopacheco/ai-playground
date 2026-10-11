import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const uiPort = Number(env.UI_PORT ?? 5303);
  const apiPort = Number(env.API_PORT ?? 8103);

  return {
    plugins: [react()],
    server: {
      port: uiPort,
      strictPort: true,
      proxy: { "/api": `http://127.0.0.1:${apiPort}` },
    },
  };
});
