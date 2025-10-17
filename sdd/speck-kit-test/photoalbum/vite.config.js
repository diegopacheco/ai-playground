import { defineConfig } from 'vite';

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: ['tests/**', '*.config.js']
    }
  },
  build: {
    target: 'es2022',
    minify: 'terser',
    sourcemap: true
  }
});
