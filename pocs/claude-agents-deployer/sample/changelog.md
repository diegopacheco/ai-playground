# Changelog

## 2026-02-09

### Built
- Added admin settings capability with global controls for comments and background theme.
- Added backend settings API and server-side enforcement for comment disable behavior.
- Added admin page in React and root-level theme application with three themes.

### Files Created
- `backend/src/models/settings.rs`
- `backend/src/handlers/settings.rs`
- `frontend/src/pages/AdminPage.tsx`
- `design-doc.md`
- `todo.md`
- `review/2026-02-09/code-review.md`
- `review/2026-02-09/sec-review.md`
- `review/2026-02-09/features.md`
- `review/2026-02-09/summary.md`

### Files Modified
- `backend/migrations/001_init.sql`
- `backend/src/handlers/comments.rs`
- `backend/src/handlers/mod.rs`
- `backend/src/main.rs`
- `backend/src/models/mod.rs`
- `backend/src/routes/api.rs`
- `backend/tests/api_integration_test.rs`
- `db/schema.sql`
- `frontend/e2e/blog.spec.ts`
- `frontend/src/hooks/useApi.ts`
- `frontend/src/index.css`
- `frontend/src/pages/PostDetailPage.tsx`
- `frontend/src/router.tsx`
- `frontend/src/types/index.ts`

### Test Coverage Summary
- `cargo check`: passed.
- `cargo test --lib`: 13 passed, 0 failed.
- `cargo test --test api_integration_test`: 0 passed, 18 failed due to DB connectivity permission restrictions.
- `npm run build`: passed.
- Playwright: blocked by package/network restrictions in sandbox.
- K6: executed but all requests failed due sandbox socket/network restrictions.

### Review Findings and Fixes
- Critical finding: settings endpoint lacks authn/authz.
- Medium findings: permissive CORS, limited mutation auditing/rate control.
- Fixes included in this change: theme allowlist validation, server-side comment-disable enforcement.

### Generated Documentation
- `design-doc.md`
- `review/2026-02-09/code-review.md`
- `review/2026-02-09/sec-review.md`
- `review/2026-02-09/features.md`
- `review/2026-02-09/summary.md`

### Remaining Issues and Recommendations
- Protect `/api/settings` with admin authentication.
- Restrict CORS for non-local environments.
- Add a local integration-test profile using isolated DB runtime when external DB is unavailable.
