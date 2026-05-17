import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    emptyOutDir: false,
    lib: {
      entry: 'frontend/src/notes-review-app.tsx',
      fileName: () => 'notes-review-app.js',
      formats: ['iife'],
      name: 'DocFlowNotesReviewApp',
    },
    minify: false,
    outDir: 'frontend/js/generated',
  },
});
