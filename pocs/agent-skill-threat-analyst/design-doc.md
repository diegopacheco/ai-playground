# Design Doc — Attack Surface Mapper Skill

## 1. Overview

A Claude Code skill that scans an entire codebase, identifies all external-facing endpoints, inputs, authentication boundaries, and produces a `threat-analysis.md` report with attack vectors, protection scores, ASCII diagrams, remediation guidance, and actionable steps.

## 2. Problem Statement

Developers ship code without a clear picture of their attack surface. Manual threat modeling is slow, inconsistent, and often skipped. This skill automates the discovery of every entry point an attacker could target and rates how well each one is protected.

## 3. Goals

- Map every external-facing endpoint (HTTP, gRPC, WebSocket, GraphQL)
- Identify all user input channels (forms, query params, headers, file uploads, CLI args)
- Detect authentication and authorization boundaries
- Classify each attack vector with a protection score (0-10)
- Generate ASCII diagrams showing the attack surface topology
- Provide concrete remediation actions for unprotected vectors
- Output everything into a single `threat-analysis.md` file

## 4. Non-Goals

- Runtime dynamic analysis (DAST) — this is static analysis only
- Dependency vulnerability scanning (use `npm audit`, `cargo audit`, etc. for that)
- Modifying any source code — read-only scan
- Replacing a full penetration test

## 5. Architecture

### 5.1 Scan Pipeline

```
Phase 1: Discovery       Phase 2: Analysis        Phase 3: Scoring        Phase 4: Report
+-----------------+     +------------------+     +-----------------+     +------------------+
| Find all source |---->| Detect endpoints |---->| Score each      |---->| Generate         |
| files, configs, |     | inputs, auth     |     | vector 0-10     |     | threat-analysis  |
| infra files     |     | boundaries       |     | with evidence   |     | .md with diagrams|
+-----------------+     +------------------+     +-----------------+     +------------------+
```

### 5.2 Detection Categories

#### A. External-Facing Endpoints

| What | Detection Strategy |
|---|---|
| REST APIs | `@GetMapping`, `@PostMapping`, `app.get(`, `router.Handle`, `@app.route`, `.route(` |
| GraphQL | `graphql`, `Query`, `Mutation`, `schema.graphql`, `typeDefs` |
| gRPC | `.proto` files, `grpc.NewServer`, `@GrpcService` |
| WebSocket | `ws://`, `wss://`, `WebSocket`, `socket.io`, `@ServerEndpoint` |
| Static file serving | `express.static`, `FileServer`, `StaticFiles` |
| Webhook receivers | `/webhook`, `/callback`, `/hook`, `@PostMapping` with external caller patterns |

#### B. Input Channels

| What | Detection Strategy |
|---|---|
| Query parameters | `req.query`, `request.args`, `r.URL.Query()`, `@RequestParam` |
| Path parameters | `req.params`, `{id}`, `:id`, `@PathVariable` |
| Request body (JSON/XML) | `req.body`, `@RequestBody`, `json.NewDecoder`, `request.json` |
| Headers | `req.headers`, `request.headers`, `r.Header.Get`, `@RequestHeader` |
| Cookies | `req.cookies`, `request.cookies`, `r.Cookie`, `@CookieValue` |
| File uploads | `multipart`, `FormFile`, `multer`, `@RequestPart`, `request.files` |
| CLI arguments | `os.Args`, `sys.argv`, `process.argv`, `clap::`, `argparse` |
| Environment variables consumed at runtime | `os.Getenv`, `process.env`, `os.environ` |
| Database inputs (indirect) | SQL queries built from external data |

#### C. Authentication Boundaries

| What | Detection Strategy |
|---|---|
| Auth middleware | `authenticate`, `authorize`, `jwt.verify`, `@PreAuthorize`, `authMiddleware` |
| Session management | `express-session`, `HttpSession`, `session`, `cookie-session` |
| OAuth/OIDC | `oauth`, `oidc`, `passport`, `spring-security-oauth` |
| API key validation | `x-api-key`, `apiKey`, `api_key` in header checks |
| RBAC/ABAC | `@RolesAllowed`, `hasRole`, `hasPermission`, `rbac`, `policy` |
| CORS configuration | `Access-Control-Allow-Origin`, `cors(`, `@CrossOrigin` |
| Rate limiting | `rateLimit`, `rate.NewLimiter`, `@RateLimiter`, `throttle` |
| CSRF protection | `csrf`, `csrfToken`, `_csrf`, `antiforgery` |

### 5.3 Scoring System

Each attack vector gets a **Protection Score** from 0 to 10:

| Score | Label | Meaning |
|---|---|---|
| 0-2 | CRITICAL | No protection found, directly exploitable |
| 3-4 | HIGH | Minimal protection, easily bypassable |
| 5-6 | MEDIUM | Some protection but gaps exist |
| 7-8 | LOW | Well protected with minor improvements possible |
| 9-10 | SECURE | Strong protection with defense in depth |

Scoring criteria per vector:

- **Input validation present?** (+2 points)
- **Auth check before processing?** (+2 points)
- **Rate limiting applied?** (+1 point)
- **Error handling does not leak info?** (+1 point)
- **Uses parameterized queries (if DB involved)?** (+1 point)
- **HTTPS/TLS enforced?** (+1 point)
- **CORS properly scoped?** (+1 point)
- **Logging/monitoring on this path?** (+1 point)

### 5.4 Attack Vector Classification

For each discovered endpoint/input, classify against OWASP Top 10 2021:

| ID | Category | What to look for |
|---|---|---|
| A01 | Broken Access Control | Missing auth on endpoints, IDOR patterns, privilege escalation paths |
| A02 | Cryptographic Failures | Weak hashing, missing encryption, exposed secrets |
| A03 | Injection | SQL injection, command injection, XSS, template injection |
| A04 | Insecure Design | Missing rate limits, no abuse prevention, business logic flaws |
| A05 | Security Misconfiguration | Debug mode, default creds, unnecessary features enabled, CORS wildcard |
| A06 | Vulnerable Components | Outdated dependencies (flag but don't deep scan) |
| A07 | Auth Failures | Weak password policy, missing MFA, session fixation |
| A08 | Data Integrity Failures | Insecure deserialization, unsigned updates |
| A09 | Logging Failures | Missing audit logs, secrets in logs |
| A10 | SSRF | Server making requests with user-supplied URLs |

## 6. Output Format — threat-analysis.md

```
# Threat Analysis Report

Generated: {date}
Codebase: {project name}
Overall Risk Score: {X}/10

## Attack Surface Diagram

{ASCII art showing:
  - External boundary
  - Each endpoint grouped by service
  - Auth gates marked with [AUTH] or [OPEN]
  - Input channels flowing in
  - Database/external service connections flowing out
}

## Executive Summary

{2-3 sentences: how many endpoints, how many unprotected,
 biggest risk areas}

## Attack Surface Inventory

### Endpoints ({count})

| # | Method | Path | Auth | Input Types | Protection Score | OWASP Risk |
|---|--------|------|------|-------------|-----------------|------------|
{one row per endpoint}

### Input Channels ({count})

| # | Channel | Location | Validated | Sanitized | Score |
|---|---------|----------|-----------|-----------|-------|
{one row per input channel}

### Auth Boundaries ({count})

| # | Mechanism | Scope | Strength | Gaps |
|---|-----------|-------|----------|------|
{one row per auth boundary}

## Threat Details

### {Vector Title} — Score: {X}/10 [{CRITICAL|HIGH|MEDIUM|LOW|SECURE}]

**Location:** `{file}:{line}`
**OWASP Category:** {A01-A10}
**Attack Scenario:**
{2-3 sentences describing how an attacker would exploit this}

**Evidence:**
{code snippet showing the vulnerable pattern}

**Current Protections:**
- {what is already in place, or "None detected"}

**Remediation:**
1. {specific action with code guidance}
2. {specific action}

**Priority:** {P0|P1|P2|P3}

---

{repeat for each vector, ordered by score ascending (worst first)}

## Protection Scorecard

| Category | Score | Status |
|----------|-------|--------|
| Input Validation | {X}/10 | {status icon} |
| Authentication | {X}/10 | {status icon} |
| Authorization | {X}/10 | {status icon} |
| Injection Prevention | {X}/10 | {status icon} |
| Cryptography | {X}/10 | {status icon} |
| Error Handling | {X}/10 | {status icon} |
| Logging & Monitoring | {X}/10 | {status icon} |
| CORS & Headers | {X}/10 | {status icon} |
| Rate Limiting | {X}/10 | {status icon} |
| Overall | {X}/10 | {verdict} |

Status: PROTECTED / PARTIAL / EXPOSED

## Action Plan

### Immediate (P0) — Fix before next deploy
{numbered list of critical fixes}

### Short Term (P1) — Fix within 1 sprint
{numbered list}

### Medium Term (P2) — Fix within 1 month
{numbered list}

### Long Term (P3) — Backlog
{numbered list}

## Appendix

### All Detected Files
{list of scanned files grouped by type}

### Scan Methodology
{brief description of what was scanned and how}
```

## 7. Diagrams

The report uses ASCII diagrams to visualize the attack surface:

### Boundary Diagram
```
                    INTERNET
                       |
            +----------+----------+
            |     LOAD BALANCER   |
            +----------+----------+
                       |
     +-----------------+-----------------+
     |                 |                 |
+----+----+      +----+----+      +----+----+
| API /v1 |      | API /v2 |      | WebSocket|
| [AUTH]  |      | [OPEN!] |      | [AUTH]   |
+----+----+      +----+----+      +----+----+
     |                 |                 |
     +--------+--------+---------+-------+
              |                  |
        +-----+-----+     +-----+-----+
        |  Database  |     |   Cache   |
        |  (Postgres)|     |  (Redis)  |
        +-----------+      +-----------+
```

### Data Flow Diagram
```
User Input --> [Validation?] --> [Auth?] --> [Handler] --> [DB Query]
                  ^                ^            |
                  |                |            v
              Missing!         Present      [Response] --> User
```

## 8. Suggestions and Improvements

### What makes this skill valuable
- Zero-config: just invoke it and it scans everything
- Actionable output: not just "you have a problem" but "here is exactly what to fix and how"
- Scored: teams can track their security posture over time by comparing scores
- Prioritized: P0/P1/P2/P3 so teams know what to fix first
- Evidence-based: every finding points to exact file:line

### Potential enhancements (future)
- Diff mode: compare two runs and show what changed
- CI integration: fail builds if score drops below threshold
- Framework-specific rules: deeper analysis for Spring Boot, Express, Django, etc.
- Custom ignore rules: let teams suppress known false positives via `.threat-ignore` file

## 9. Critiques and Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Static analysis only | Cannot detect runtime-only vulnerabilities (timing attacks, race conditions) | Clearly state this in report, recommend DAST tools for complement |
| Pattern matching is imperfect | May miss custom auth frameworks or non-standard patterns | Use heuristics + code reading for context, flag low-confidence findings |
| No dependency CVE scanning | Misses known vulnerabilities in third-party packages | Recommend `npm audit`, `cargo audit`, `mvn dependency-check` in report |
| Large codebases slow to scan | Timeout risk on very large monorepos | Chunk scanning, skip vendor/generated code |
| Language coverage gaps | Some languages or frameworks may not be covered | Start with Java, Go, Rust, Python, JS/TS — expand over time |
| False positives | Internal-only endpoints flagged as external | User must review findings, skill marks confidence level |
| Cannot verify runtime config | Environment variables may change protection at deploy time | Flag env-dependent security controls, recommend checking prod config |

## 10. File Structure

```
~/.claude/skills/threat-analyst/
  SKILL.md          -- The skill definition
agent-skill-threat-analyst/
  design-doc.md     -- This document
  install.sh        -- Copies SKILL.md to ~/.claude/skills/threat-analyst/
  uninstall.sh      -- Removes ~/.claude/skills/threat-analyst/
```

## 11. Invocation

```
/threat-analyst
/threat-analyst src/api/
/threat-analyst --focus auth
```

The skill accepts optional arguments to narrow the scan scope.
