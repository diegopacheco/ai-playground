import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

function required(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);

  return value;
}

const web = new URL("./web", import.meta.url).pathname;

export default defineConfig({
  root: web,
  plugins: [tanstackRouter({ target: "react", routesDirectory: `${web}/src/routes`, generatedRouteTree: `${web}/src/routeTree.gen.ts` }), react()],
  server: {
    port: Number(required("WEB_PORT")),
    strictPort: true,
    proxy: { "/api": `http://localhost:${required("API_PORT")}` },
  },
});
