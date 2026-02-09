# Change Summary - 2026-02-09

## What Was Built
- Added global admin settings persisted in database.
- Added admin UI to manage comment availability and background theme.
- Added backend enforcement so comments cannot be created when disabled.
- Added three UI background themes.

## Files Added
- `backend/src/models/settings.rs`
- `backend/src/handlers/settings.rs`
- `frontend/src/pages/AdminPage.tsx`
- `design-doc.md`
- `todo.md`

## Major Files Updated
- `backend/src/main.rs`
- `backend/src/routes/api.rs`
- `backend/src/handlers/comments.rs`
- `backend/tests/api_integration_test.rs`
- `backend/migrations/001_init.sql`
- `db/schema.sql`
- `frontend/src/router.tsx`
- `frontend/src/pages/PostDetailPage.tsx`
- `frontend/src/hooks/useApi.ts`
- `frontend/src/types/index.ts`
- `frontend/src/index.css`
- `frontend/e2e/blog.spec.ts`

## Validation Snapshot
- Backend compile: passed.
- Frontend build: passed.
- Backend unit tests: passed.
- Integration/UI/K6: blocked by sandbox network and DB connectivity constraints.
