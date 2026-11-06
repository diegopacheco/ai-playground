import { readFileSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const ports = Object.fromEntries(
  readFileSync(new URL("../scripts/ports.env", import.meta.url), "utf8")
    .split("\n")
    .filter((line) => line.includes("="))
    .map((line) => line.trim().split("=")),
);

export default defineConfig({
  plugins: [react()],
  define: { __COMPANY_URL__: JSON.stringify(`http://localhost:${ports.backend}`) },
  server: { host: "127.0.0.1", port: Number(ports.ui), strictPort: true },
});
