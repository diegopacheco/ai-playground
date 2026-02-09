# Security Review - 2026-02-09

## High
- `backend/src/routes/api.rs` exposes admin settings update (`PUT /api/settings`) without authentication or authorization. Any client can disable comments or change theme.

## Medium
- `backend/src/main.rs` still configures permissive CORS (`allow_origin(Any)`, `allow_methods(Any)`, `allow_headers(Any)`), expanding attack surface for browser-based misuse.
- `backend/src/handlers/settings.rs` has no rate limit or audit trail for settings mutation.

## Low
- Theme values are validated server-side against a fixed allowlist, which is good. Input validation scope remains narrow and does not yet cover request size limits.

## Summary
The new feature has server-side enforcement for comment blocking and strict theme allowlist checks, but admin endpoint protection is the primary missing control.
