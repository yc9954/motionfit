import { defineConfig } from "vite";
import path from "path";

export default defineConfig({
  root: ".",
  base: "/",
  build: {
    outDir: path.resolve(__dirname, "../static"),
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8080",
        changeOrigin: true,
      },
    },
  },
});
