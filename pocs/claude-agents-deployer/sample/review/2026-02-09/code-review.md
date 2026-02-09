# Code Review - 2026-02-09

## High
- `k6/run-tests.sh` runs long sequences even when the API is unreachable. In constrained environments this creates noisy, long-running failures.

## Medium
- `frontend/src/router.tsx` fetches settings at the root layout but does not show loading or error state for settings retrieval. Theme falls back silently.
- `backend/src/handlers/settings.rs` allows partial updates but does not validate payload emptiness. Empty payload writes `updated_at` without effective config change.

## Low
- `frontend/src/pages/PostDetailPage.tsx` hides comment form when comments are disabled, but no explicit refresh trigger exists if settings are changed while page is open.
- `backend/tests/api_integration_test.rs` relies on external PostgreSQL availability; no fallback test mode is present for restricted CI environments.

## Test Coverage Notes
- Backend model unit tests passed.
- Integration tests include new settings flows and comment-disable enforcement, but execution depends on DB access.
