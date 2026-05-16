import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    emptyOutDir: false,
    lib: {
      entry: 'frontend/src/settings-app.tsx',
      fileName: () => 'settings-app.js',
      formats: ['iife'],
      name: 'DocFlowSettingsApp',
    },
    minify: false,
    outDir: 'frontend/js/generated',
  },
});
