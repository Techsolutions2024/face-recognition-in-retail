## Face Recognition Web Migration Notes

### Architecture Overview
- **Backend (`backend/`)**
  - FastAPI (`backend/app.py`) serving REST + Server-Sent Event stream.
  - SQLAlchemy models in `backend/database.py` map existing SQLite schema.
  - Authentication with JWT + bcrypt hashing (`backend/security.py`).
  - Feature routers in `backend/api/routes/` covering auth, users, customers, cameras, events, crops, settings, stats.
  - Static media served at `/media/crops` and `/media/gallery`.
  - Realtime event feed via `/api/v1/events/stream`.
  - Tests located in `backend/tests`.

- **Frontend (`web/`)**
  - Vite + React + TypeScript + Ant Design + Tailwind.
  - State handled with React Query + Zustand (`src/store/authStore.ts`).
  - Routing defined in `src/routes/AppRoutes.tsx`.
  - Admin pages under `src/pages/admin/` (dashboard, events, crops, customers, cameras, users, settings).
  - Client display at `src/pages/client/ClientOverviewPage.tsx`.
  - Event stream hook (`src/hooks/useEventStream.ts`) consumes SSE and updates caches.

### Running Locally
1. **Backend**
   ```bash
   uvicorn backend.app:app --reload --port 8000
   ```
   - Ensure `requirements.txt` dependencies installed.
   - Environment overrides via `.env` (see `backend/config.py` for keys).

2. **Frontend**
   ```bash
   cd web
   npm install
   npm run dev
   ```
   - Vite dev server proxies `/api/*` to `http://localhost:8000`.
   - Tailwind + Ant Design ready for custom styling.

### Media & Recognition Pipeline
- Existing OpenVINO pipeline remains in `face_module` for now.
- Backend exposes `crops_directory` and `gallery_directory` configurable in `.env`.
- Future work: extract recognition worker into service that emits events into DB or message broker, consumed by API for realtime streaming.

### Testing
- Backend tests runnable via `pytest backend/tests`.
- Recommended: add Playwright/Cypress for end-to-end flows once the web client connects to live backend.

### Outstanding Enhancements
- Integrate real video stream (WebRTC/HLS) in client.
- Expand SSE to push crop-only updates and user notifications.
- Harden auth (refresh rotation, password reset, RBAC) and introduce rate limiting.
- Document data migration steps for existing deployments.

