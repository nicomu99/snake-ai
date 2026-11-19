import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The FastAPI backend (uvicorn) runs on :8000. Proxying /ws through the Vite
// server lets the app always connect to its own origin, so the same code works
// in dev and when FastAPI serves the built files later.
const backend = {
  '/ws': { target: 'http://localhost:8000', ws: true },
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { proxy: backend },
  preview: { proxy: backend },
})
