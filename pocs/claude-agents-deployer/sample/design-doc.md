# Design Doc

## Architecture Overview
The system keeps a single global application settings record in PostgreSQL and exposes it through backend REST endpoints. The React frontend reads these settings to control visual theme and comment behavior. An admin page allows updating settings. Comment creation is enforced by the backend to prevent bypass.

## Backend API Endpoints and Responsibilities
- `GET /api/settings`
Returns global settings with `commentsEnabled` and `backgroundTheme`.

- `PUT /api/settings`
Updates global settings. Accepted themes: `classic`, `forest`, `sunset`.

- `POST /api/posts/{post_id}/comments`
Now validates global settings before creating comments. Returns `403` when comments are disabled.

## Frontend Components and Interactions
- `AdminPage`
Provides controls to enable or disable comments and select one of three themes.

- Root layout
Loads settings and applies selected theme to the app wrapper via `data-theme`.

- `PostDetailPage`
Reads settings and disables comment submission UI when comments are disabled.

## Database Schema Design
A new `settings` table stores a single row:
- `id INT PRIMARY KEY` with value `1`
- `comments_enabled BOOLEAN NOT NULL`
- `background_theme TEXT NOT NULL` with check constraint for allowed values
- `updated_at TIMESTAMP NOT NULL`

Initialization inserts default row when absent.

## Integration Points Between Frontend, Backend, Database
- Frontend admin updates settings through `/api/settings`.
- Backend persists settings in PostgreSQL.
- Frontend reads settings from `/api/settings` for runtime rendering.
- Backend comment creation checks persisted setting before write.
