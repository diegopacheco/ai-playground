# Security Review - 2026-02-12

## Summary

The application has a solid security baseline with proper password hashing (Argon2), JWT authentication, parameterized SQL queries, and input validation. The critical finding is the hardcoded JWT secret. Other findings are moderate to low severity.

---

## Critical Findings

### SEC-01: Hardcoded JWT Secret Key
**File**: `backend/src/auth.rs` line 10
**Severity**: CRITICAL
**OWASP**: A07:2021 - Identification and Authentication Failures

The JWT signing key is hardcoded as a constant:
```
const JWT_SECRET: &str = "twitter-clone-secret-key-change-in-production";
```

This means:
- The secret is visible in the source code repository
- All deployments share the same secret
- Anyone with access to the source can forge valid JWTs for any user
- The secret cannot be rotated without redeploying

**Recommendation**: Load the JWT secret from an environment variable at startup. Fail fast if the variable is not set in production.

---

## High Severity Findings

### SEC-02: No Rate Limiting on Authentication Endpoints
**File**: `backend/src/routes/auth.rs`
**Severity**: HIGH
**OWASP**: A07:2021 - Identification and Authentication Failures

No rate limiting on `/api/auth/login` allows brute-force password attacks. No rate limiting on `/api/auth/register` allows mass account creation.

**Recommendation**: Add rate limiting middleware (e.g., `tower-governor`) to auth endpoints. Limit login to 5 attempts per IP per minute and registration to 3 per IP per 10 minutes.

### SEC-03: No Password Strength Requirements
**File**: `backend/src/routes/auth.rs` line 16
**Severity**: HIGH
**OWASP**: A07:2021 - Identification and Authentication Failures

The registration endpoint only checks that the password is non-empty. Users can register with single-character passwords like "a".

**Recommendation**: Enforce minimum password length (at least 8 characters) and consider checking against common password lists.

---

## Medium Severity Findings

### SEC-04: CORS Allows Any Methods and Headers
**File**: `backend/src/main.rs` lines 20-23
**Severity**: MEDIUM
**OWASP**: A05:2021 - Security Misconfiguration

While the origin is correctly restricted to `http://localhost:5173`, methods and headers use `Any`. This is overly permissive. Only GET, POST, DELETE methods and Content-Type, Authorization headers are actually needed.

### SEC-05: JWT Token Has 24-Hour Expiration With No Refresh Mechanism
**File**: `backend/src/auth.rs` line 14
**Severity**: MEDIUM
**OWASP**: A07:2021 - Identification and Authentication Failures

Tokens are valid for 24 hours with no refresh token mechanism and no revocation capability. If a token is compromised, it remains valid for up to 24 hours.

**Recommendation**: Consider shorter token lifetimes (1 hour) with a refresh token endpoint, or implement a token blacklist for logout.

### SEC-06: No CSRF Protection
**Severity**: MEDIUM
**OWASP**: A01:2021 - Broken Access Control

The application uses JWT in the Authorization header, which provides implicit CSRF protection since browsers do not automatically send this header. However, if the token storage mechanism changes (e.g., to cookies), CSRF would become a concern. The current localStorage approach is acceptable.

### SEC-07: Foreign Keys Not Enforced at Runtime
**File**: `backend/src/db.rs`
**Severity**: MEDIUM
**OWASP**: A04:2021 - Insecure Design

SQLite foreign keys are disabled by default. The backend does not execute `PRAGMA foreign_keys = ON`. This means a user could potentially manipulate requests to reference non-existent users or posts in follows/likes without the database rejecting them (though the application code does check existence in most cases).

---

## Low Severity Findings

### SEC-08: User Email Exposed in Profile Response
**File**: `backend/src/routes/users.rs` line 39
**Severity**: LOW
**OWASP**: A01:2021 - Broken Access Control

The `get_user` endpoint returns the user's email address to any authenticated user. Email addresses are personally identifiable information and their exposure could enable targeted phishing.

### SEC-09: Error Messages May Leak Implementation Details
**File**: `backend/src/errors.rs` lines 30-48
**Severity**: LOW
**OWASP**: A09:2021 - Security Logging and Monitoring Failures

Database errors, JWT errors, and password errors are logged with `tracing::error!` including the full error details, but the response only returns generic messages. This is correct behavior. However, the `register` handler returns "Username or email already exists" which allows user enumeration (attacker can determine if an email is registered).

### SEC-10: Token Stored in localStorage
**File**: `frontend/src/auth.ts`
**Severity**: LOW
**OWASP**: A07:2021 - Identification and Authentication Failures

JWT tokens stored in localStorage are accessible to any JavaScript running on the page. If an XSS vulnerability exists, the token can be exfiltrated. The React framework provides built-in XSS protection for rendered content, but any future `dangerouslySetInnerHTML` usage or third-party script inclusion could compromise tokens.

---

## Positive Security Practices

1. **Password Hashing**: Argon2 with random salt generation per user
2. **Parameterized Queries**: All SQL uses bind parameters (`?1`, `?2`), preventing SQL injection
3. **No Raw HTML Rendering**: React JSX escapes content by default, preventing XSS
4. **Authorization Checks**: Post deletion verifies ownership, self-follow prevention exists
5. **Proper Error Responses**: Internal errors return generic messages, not stack traces
6. **Password Hash Exclusion**: `UserResponse` type omits `password_hash` field
7. **Auth Middleware**: Centralized authentication check via Axum middleware layer
8. **Input Validation**: Email format, post length, empty field checks on backend
