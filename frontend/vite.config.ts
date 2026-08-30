import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // 0.0.0.0 — reachable on LAN (phone / other PCs)
    port: 5173,
    strictPort: true,
    proxy: {
      "/health": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
      },
      "/v1": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
      },
    },
  },
});
