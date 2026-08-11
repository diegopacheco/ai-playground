import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "./src"),
      "@/components": resolve(__dirname, "./src/components"),
      "@/hooks": resolve(__dirname, "./src/hooks"),
      "@/services": resolve(__dirname, "./src/services"),
      "@/types": resolve(__dirname, "./src/types"),
      "@/styles": resolve(__dirname, "./src/styles"),
      "@/pages": resolve(__dirname, "./src/pages"),
    },
  },
  server: {
    port: 5173,
    open: false,
  },
  preview: {
    port: 5173,
  },
});
