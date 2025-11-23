# Face Recognition UI Audit

## Overview
This document captures the current PyQt-based UX in `face_module` to guide the React refactor.

## Login Flow (`login_window.py`)
- Desktop-only animated split layout with product marketing content and login card.
- Authenticates via `Database.authenticate_user(username, password)`.
- Supports "remember me" UI flag (no persisted storage yet).
- Shows pre-filled demo credentials for admin `admin/1234` and client `client/1234`.
- Emits `login_successful` signal with `{'id', 'username', 'role'}` used by `MainApplication`.

## Main Application (`main.py`)
- Initializes SQLite database `facere.db` through `Database` class.
- Displays `LoginWindow` and routes to `AdminPanel` or `ClientPanel` based on user role.
- Keeps single instance of each panel with `logout_signal` returning to login.

## Admin Panel (`admin_panel.py`)
### Dashboard
- Header gradient bar with user info, logout.
- Tabs: Dashboard, Events, Crops, Management.
- Realtime refresh timer (5s) to update stats from `Database` (`get_event_count_today`, `get_event_stats`, `get_recent_events`).
- Widgets:
  - KPI cards (recognized VIPs today, unknown detections, active cameras, dwell time, new customers).
  - Timeline list of recent VIP events with detail modal showing crop image and metadata.
  - Active visits table (customers currently in store).
  - Camera status grid with per-camera actions (activate/deactivate, edit source).

### Events Tab
- Paginated table (20/page) of events including filters by type, date range, camera.
- `EventDetailDialog` for full metadata and associated crops.

### Crops Tab
- Scrollable gallery of captured face crops with metadata (customer, confidence, timestamp).
- Pagination (24/page) with lazy image loading from `./crops` folder.

### Management Tab
- Sub-tabs for Users, Customers, Cameras, Settings, Models.
- User management invokes `UserManagementDialog` (CRUD on `users`).
- Customer management invokes `CustomerManagementDialog` (CRUD on `customers`, optional gallery uploads, segmentation).
- Camera management allows add/edit/delete camera sources stored in `cameras` table.
- Settings forms to adjust recognition thresholds, crop capture interval (`settings` table), gallery directory, etc.
- Model management to load OpenVINO models (`utils.FaceDetector`, etc.), trigger recalibration, rebuild descriptors.
- Admin-only ability to start/stop live recognition pipeline and push model updates to client panel via shared DB flags.

### Recognition Pipeline
- Embedded OpenVINO initialization (`Core`, `FaceDetector`, `LandmarksDetector`, `FaceIdentifier`, `FacesDatabase`).
- Video capture (RTSP/IP/webcam) with face detection, tracking, and crop storage via `CropsManager`.
- On recognition:
  - Logs event via `Database.add_event` with `EventType`.
  - Saves crop to disk and DB (`add_crop`).
  - Updates customer visit stats (`increment_customer_visits`, `add_visit_entry`).
  - Emits in-panel notifications and audio cues (not yet abstracted for web).

## Client Panel (`client_panel.py`)
- Read-only layout with toolbar (user info, start/stop, logout), live video stream, and side panel of recent detections.
- Uses same recognition thread as Admin but without management actions; fetches model paths from DB settings.
- Displays:
  - Live camera feed with bounding boxes and labels.
  - FPS indicator and status bar.
  - Recent detections list of crops + metadata (timestamp, confidence).
- Timer checks for model config changes to reload models when admin updates settings.

## Supporting Dialogs
- `UserManagementDialog`: CRUD for authentication users, validations, prevents self-deletion.
- `CustomerManagementDialog`: CRUD with optional image upload -> processed by recognition models to add descriptors to gallery.
- `CropsManager`, `EventsManager`: background helpers for saving crops and summarizing events.

## Database Layer (`database.py`)
- SQLite schema: `users`, `customers`, `events`, `crops`, `visits`, `cameras`, `settings`.
- Provides numerous query helpers for stats, filters, pagination, and configuration values (e.g., capture interval, model paths).
- Responsible for default admin/client user seeding.

## Key Feature Requirements for React Refactor
- Auth with roles (admin/client) and session persistence.
- Admin dashboard replicating current KPIs, charts, timelines, and camera/visit insights.
- CRUD management for users, customers, cameras, settings, models (file uploads for gallery + model binaries).
- Real-time events and crops feed (WebSocket/SSE) with modal details.
- Video streaming integration (from backend service or WebRTC-proxied feed) for live monitoring.
- Client portal limited to viewing live recognitions and recent events.
- Support for uploading face images and storing to gallery, with backend processing.
- Configuration updates (thresholds, capture interval) applied system-wide.

## Gaps & Web Considerations
- Need to externalize recognition pipeline into backend service (headless) instead of PyQt threads.
- Video streaming must be browser-compatible (HLS, WebRTC, MJPEG) rather than OpenCV window.
- File storage paths for crops/gallery should be served via HTTP with proper access controls.
- Replace PyQt timers/signals with REST + WebSocket messaging.
- Introduce proper authentication (JWT) and password hashing (current DB stores plaintext).
- Align database migrations (consider SQLAlchemy + Alembic) for maintainability.
