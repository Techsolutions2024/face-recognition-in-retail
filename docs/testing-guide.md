## Testing Guide

### Backend (FastAPI)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run unit tests:
   ```bash
   pytest backend/tests
   ```
3. Manual verification:
   - Start API: `uvicorn backend.app:app --reload --port 8000`
   - Exercise endpoints with `httpie` or `curl`, e.g. `curl -X POST http://localhost:8000/api/v1/auth/token -d '{"username":"admin","password":"1234"}' -H "Content-Type: application/json"`

### Frontend (React / Vite)
1. Install packages:
   ```bash
   cd web
   npm install
   ```
2. Run linter & type-check:
   ```bash
   npm run lint
   npx tsc --noEmit
   ```
3. Launch dev server:
   ```bash
   npm run dev
   ```
   - Access admin console at `http://localhost:5173/admin/dashboard`
   - Access client monitor at `http://localhost:5173/client`

### Integration Checklist
- Verify login flow for both admin and client accounts.
- Confirm dashboard metrics update when new events are inserted into SQLite database.
- Validate CRUD flows for customers, cameras, users, settings.
- Confirm SSE stream (`/api/v1/events/stream`) pushes new events to React dashboards.
- Ensure media files under `crops/` are accessible via `http://localhost:8000/media/crops/...`.

