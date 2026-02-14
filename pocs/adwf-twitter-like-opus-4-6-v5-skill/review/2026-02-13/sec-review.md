# Security Review - 2026-02-13

## OWASP Top 10 Analysis

### A01 - Broken Access Control: LOW RISK
- Protected routes use JWT middleware that validates tokens before handler execution
- Tweet deletion checks ownership (user_id in WHERE clause)
- Follow/unfollow operations use the authenticated user's ID from the JWT
- Self-follow prevented by database CHECK constraint

### A02 - Cryptographic Failures: LOW RISK
- Passwords hashed with BCrypt (cost factor 10)
- JWT uses HS256 with a configurable secret
- Password hash never exposed in UserResponse (stripped via From<User> implementation)
- Recommendation: Use a stronger JWT secret in production (env var JWT_SECRET)

### A03 - Injection: LOW RISK
- All SQL queries use parameterized queries via SQLx ($1, $2 placeholders)
- No string concatenation in SQL queries
- No shell command execution
- Frontend uses axios which properly encodes request parameters

### A04 - Insecure Design: MEDIUM RISK
- No rate limiting on login/register endpoints (brute force possible)
- No account lockout after failed login attempts
- No CAPTCHA on registration
- Recommendation: Add rate limiting middleware

### A05 - Security Misconfiguration: LOW RISK
- CORS configured to specific origin (localhost:5173)
- However, allow_methods and allow_headers use `Any` which is overly permissive
- Recommendation: Restrict to specific HTTP methods and headers

### A06 - Vulnerable Components: LOW RISK
- Dependencies are current versions (Axum 0.8, SQLx 0.8, React 19)
- No known vulnerabilities in used crate versions

### A07 - Authentication Failures: LOW RISK
- JWT tokens expire after 24 hours
- Invalid/expired tokens properly rejected with 401
- Password verification uses constant-time comparison (BCrypt)
- No token refresh mechanism (acceptable for this scope)

### A08 - Data Integrity Failures: LOW RISK
- Database constraints enforce data integrity
- Foreign keys prevent orphaned records
- UNIQUE constraints prevent duplicate likes/follows

### A09 - Logging Failures: MEDIUM RISK
- Tracing is initialized but not extensively used in handlers
- Failed login attempts are not logged
- Recommendation: Add structured logging for auth events

### A10 - SSRF: NOT APPLICABLE
- No server-side URL fetching

## JWT Implementation
- Token creation uses jsonwebtoken crate with HS256
- Claims include user_id (sub) and expiration (exp)
- Token validation checks signature and expiration
- Secret is configurable via JWT_SECRET environment variable (defaults to "super-secret-jwt-key")
- Recommendation: Change default secret in production

## Input Validation
- Tweet content validated: non-empty and max 280 characters
- Email uniqueness enforced at database level
- Username uniqueness enforced at database level
- No explicit email format validation (relies on client-side)

## Frontend Security
- JWT stored in localStorage (XSS risk, but standard for SPAs)
- No sensitive data in console logs
- API client uses axios interceptor for consistent auth headers

## Summary
No critical vulnerabilities found. The application follows security best practices for its scope. Key recommendations: add rate limiting, restrict CORS methods/headers, and add logging for auth events.
