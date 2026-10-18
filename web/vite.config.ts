import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development Vite serves the screen and forwards /api to the server (docker/dev.sh starts both).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
    // IP addresses are always allowed; this also lets in LAN names such as mars.local.
    allowedHosts: [".local"],
  },
});
