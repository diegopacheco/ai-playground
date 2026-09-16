import { defineConfig } from "vite";

export default defineConfig({
  build: {
    target: "es2023",
    lib: { entry: "src/index.ts", formats: ["es"], fileName: () => "index.js" },
    minify: false,
    emptyOutDir: true
  }
});
