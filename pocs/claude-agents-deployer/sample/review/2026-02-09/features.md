# Feature Documentation - 2026-02-09

## Added Feature
Simple admin panel for global platform controls:
- Toggle comments on or off for all posts.
- Select one of three background themes: `classic`, `forest`, `sunset`.

## Backend
- New settings model: `backend/src/models/settings.rs`
- New settings handler: `backend/src/handlers/settings.rs`
- New API endpoints:
  - `GET /api/settings`
  - `PUT /api/settings`
- Comment creation now checks global settings and returns `403` when comments are disabled.

## Database
- Added `settings` table with constrained `background_theme` values.
- Added seed row (`id=1`) for initial state.
- Updated DB schema files:
  - `backend/migrations/001_init.sql`
  - `db/schema.sql`

## Frontend
- New admin page: `frontend/src/pages/AdminPage.tsx`
- New route and navigation link: `/admin`
- Theme application at app root via `data-theme`.
- Post details page now respects comments-enabled setting and hides add-comment form when disabled.
- New settings hooks and types in:
  - `frontend/src/hooks/useApi.ts`
  - `frontend/src/types/index.ts`
