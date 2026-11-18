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
  define: { "import.meta.env.VITE_RESTATE_ADMIN": JSON.stringify(`http://localhost:${ports["restate-admin"]}`) },
  server: {
    proxy: {
      "/ingress": { target: `http://localhost:${ports["restate-ingress"]}`, rewrite: path => path.replace(/^\/ingress/, "") },
      "/admin": { target: `http://localhost:${ports["restate-admin"]}`, rewrite: path => path.replace(/^\/admin/, "") },
    },
  },
});
