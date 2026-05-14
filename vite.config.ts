import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    emptyOutDir: true,
    lib: {
      entry: 'frontend/src/stream-parser.ts',
      fileName: () => 'stream-parser.js',
      formats: ['iife'],
      name: 'DocFlowStreamParser',
    },
    minify: false,
    outDir: 'frontend/js/generated',
  },
  test: {
    environment: 'node',
    include: ['frontend/tests/**/*.test.ts'],
  },
});

