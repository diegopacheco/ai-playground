import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { localCredentials } from "./credentials-plugin";

const ports = Object.fromEntries(
  readFileSync(resolve(__dirname, "../../scripts/ports.env"), "utf8")
    .split("\n")
    .filter((line) => line.includes("="))
    .map((line) => line.split("=").map((part) => part.trim())),
);

const target = (name: string) => `http://127.0.0.1:${ports[name]}`;
const proxy = (name: string) => ({
  target: target(name),
  rewrite: (path: string) => path.replace(new RegExp(`^/api/${name}`), ""),
});

export default defineConfig({
  plugins: [react(), localCredentials(resolve(__dirname, "../.secrets"), target("auth"))],
  server: {
    proxy: {
      "/api/auth": proxy("auth"),
      "/api/state": proxy("state"),
      "/api/logs": proxy("logs"),
      "/api/evals": proxy("evals"),
    },
  },
});
