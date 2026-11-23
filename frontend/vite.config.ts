import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The FastAPI backend (uvicorn) runs on :8000. Proxying the backend routes
// through the Vite server lets the app always talk to its own origin, so the
// same code works in dev and when FastAPI serves the built files later.
// Every backend route must be listed here: Vite answers any other path with
// index.html (status 200), so a missing entry shows up as a JSON parse error.
const backend = {
  '/ws': { target: 'http://localhost:8000', ws: true },
  '/models': 'http://localhost:8000',
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { proxy: backend },
  preview: { proxy: backend },
})
