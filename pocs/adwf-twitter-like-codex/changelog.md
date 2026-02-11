# Changelog

## 2026-02-11

### Built
- Backend project in `backend/` with Axum API routes for users, follows, posts, likes, and timeline.
- Frontend app in `frontend/` with React UI for creating users/posts, listing posts, and liking posts.
- Database schema in `db/schema.sql` and backend migration `backend/migrations/001_init.sql`.
- Test scripts in `tests/` for integration, UI, and stress execution.

### Files Created or Modified
- `design-doc.md`
- `todo.md`
- `db/schema.sql`
- `backend/Cargo.toml`
- `backend/migrations/001_init.sql`
- `backend/src/main.rs`
- `frontend/index.html`
- `frontend/package.json`
- `frontend/src/main.jsx`
- `frontend/src/styles.css`
- `frontend/tests/ui.spec.ts`
- `tests/integration.sh`
- `tests/ui.sh`
- `tests/k6.js`
- `tests/stress.sh`
- `review/2026-02-11/code-review.md`
- `review/2026-02-11/sec-review.md`
- `review/2026-02-11/features.md`
- `review/2026-02-11/summary.md`
- `README.md`

### Test Coverage Summary
- Unit tests: `cd backend && cargo test` passed (2 tests).
- Integration tests: `./tests/integration.sh` passed.
- UI tests: `./tests/ui.sh` passed (1 Playwright test).
- Stress tests: `./tests/stress.sh` passed with status checks successful.

### Review Findings and Fixes
- No critical findings.
- Noted medium risks: no auth, open CORS, fixed frontend API host.

### Documentation Generated
- `design-doc.md`
- `review/2026-02-11/code-review.md`
- `review/2026-02-11/sec-review.md`
- `review/2026-02-11/features.md`
- `review/2026-02-11/summary.md`
- `README.md`

### Remaining Recommendations
- Add authentication and authorization.
- Move frontend API host to environment configuration.
- Add pagination and request rate limiting.
