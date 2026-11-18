import { readFileSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const ports = Object.fromEntries(
  readFileSync(new URL("../scripts/ports.env", import.meta.url), "utf8")
    .split("\n")
    .filter(line => line.includes("="))
    .map(line => line.split("=").map(part => part.trim())),
);

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: `http://localhost:${ports["agent-api"]}`, rewrite: path => path.replace(/^\/api/, "") },
    },
  },
});
