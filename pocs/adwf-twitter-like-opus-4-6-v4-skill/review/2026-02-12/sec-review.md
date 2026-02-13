# Security Review - Twitter-like App

**Date**: 2026-02-12

## OWASP Top 10 Analysis

### A01:2021 - Broken Access Control

- **Post deletion**: Only the post owner can delete their post. The handler checks `post.user_id != auth.user_id`. This is correctly enforced.
- **Follow/Unfollow**: Self-follow is prevented both at the application level (`auth.user_id == user_id` check) and at the database level (`CHECK(follower_id != following_id)`).
- **Feed**: Only shows posts from followed users and the authenticated user's own posts. Access control is correct.
- **Missing**: `get_followers` and `get_following` endpoints are public (no auth required). User social graphs are visible to unauthenticated users.
- **Missing**: `get_all_posts` and `get_post` are public. All post content is accessible without authentication.

### A02:2021 - Cryptographic Failures

- **Critical: Hardcoded JWT secret**: The JWT secret is hardcoded as `"twitter-like-super-secret-key-2024"` in `auth.rs` line 8. This secret is committed to source control. In production, this must be loaded from an environment variable or secrets manager.
- **bcrypt**: Password hashing uses bcrypt with cost factor 12, which is adequate. Passwords are never logged or returned in API responses.

### A03:2021 - Injection

- **SQL Injection**: All SQL queries use parameterized bindings (`?` placeholders with `.bind()`). No string concatenation is used in query construction. SQL injection risk is mitigated.
- **XSS**: The backend serves only JSON. No HTML rendering occurs server-side. The React frontend handles rendering, and React auto-escapes JSX output by default. The `whitespace-pre-wrap` CSS class preserves whitespace but does not interpret HTML.

### A04:2021 - Insecure Design

- No rate limiting on login, register, or post creation endpoints. An attacker can brute-force credentials or spam posts.
- No account lockout mechanism after failed login attempts.
- No email verification on registration. Any email format is accepted.

### A05:2021 - Security Misconfiguration

- **CORS**: Configured to allow only `http://localhost:5173` as origin, which is appropriate for development. However, `allow_methods(Any)` and `allow_headers(Any)` are overly permissive. In production, these should be restricted to the specific methods and headers needed.
- The server binds to `0.0.0.0:8080`, meaning it accepts connections from any network interface.

### A06:2021 - Vulnerable and Outdated Components

- Dependencies use latest major versions (Axum 0.8, SQLx 0.8, jsonwebtoken 9, bcrypt 0.17). No known vulnerabilities at the time of review.

### A07:2021 - Identification and Authentication Failures

- JWT tokens expire after 24 hours, which is reasonable.
- No token refresh mechanism. Users must re-login after 24 hours.
- No token revocation mechanism. A compromised token remains valid until expiration.
- Login returns generic "Invalid credentials" for both wrong username and wrong password, which is correct to prevent user enumeration.

### A08:2021 - Software and Data Integrity Failures

- JWT uses HS256 (HMAC-SHA256) via `Header::default()`. The algorithm is not explicitly set, relying on the library default. This is acceptable but explicit algorithm specification would be more secure.
- No CSRF protection, though this is less critical for a token-based (not cookie-based) authentication scheme.

### A09:2021 - Security Logging and Monitoring Failures

- No structured logging. The only log output is `println!("Server running on http://0.0.0.0:8080")`.
- Failed login attempts are not logged.
- No audit trail for user actions (post creation, deletion, follows).

### A10:2021 - Server-Side Request Forgery (SSRF)

- Not applicable. The backend does not make outbound HTTP requests based on user input.

## JWT Security Details

| Aspect | Status |
|--------|--------|
| Secret management | CRITICAL: Hardcoded in source code |
| Algorithm | HS256 (default) |
| Expiration | 24 hours |
| Token refresh | Not implemented |
| Token revocation | Not implemented |
| Claims | sub (user_id), username, exp |

## Input Validation

| Endpoint | Validation |
|----------|-----------|
| Register | Empty field check only. No username length limit, no email format validation, no password strength requirements. |
| Login | No validation beyond matching credentials. |
| Create Post | Empty check and 280 character max. Content length also enforced at database level via CHECK constraint. |
| Follow | Self-follow prevented. Target user existence verified. |
| Like | Post existence verified before insert. |

## Sensitive Data Exposure

- `UserResponse` correctly excludes `password_hash`. The `From<User>` conversion omits it.
- Internal SQLx error messages are forwarded to clients via `AppError::Internal(err.to_string())`. This can leak database schema details.
- The JWT token is stored in `localStorage` on the frontend, which is vulnerable to XSS attacks if any XSS vulnerability exists. HttpOnly cookies would be more secure.

## Summary of Critical Findings

1. **Hardcoded JWT secret** in source code
2. **No rate limiting** on any endpoint
3. **No input sanitization** on registration (no email format check, no password policy)
4. **Internal error messages** exposed to clients
5. **Token stored in localStorage** instead of HttpOnly cookies
6. **No security logging** for authentication events
