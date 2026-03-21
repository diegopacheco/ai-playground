# Operational Runbook — Agents Auction House

Generated: 2026-03-20
Completeness Score: 100% (7 of 7 detected components covered)

## Service Overview

### Architecture Summary
A two-tier web application where AI coding agents (Claude, Gemini, Copilot, Codex) compete in sequential auctions. The Go backend (Gin) runs on port 3000, manages auction state in SQLite, orchestrates agent CLI invocations with 20-second timeouts, and streams real-time events via SSE. The React/Vite frontend on port 5173 proxies API calls to the backend and renders live auction progress.

### Component Inventory
| Component | Type | Location | Port |
|---|---|---|---|
| Go Backend (Gin) | HTTP Server | `backend/main.go` | 3000 |
| React Frontend (Vite) | SPA Dev Server | `frontend/vite.config.ts` | 5173 |
| SQLite Database | Embedded DB | `backend/auction.db` (WAL mode) | N/A |
| SSE Broadcaster | In-process pub/sub | `backend/sse/broadcaster.go` | 3000 (via `/api/auctions/:id/stream`) |
| Claude CLI Agent | External CLI process | `backend/agents/claude.go` | N/A |
| Gemini CLI Agent | External CLI process | `backend/agents/gemini.go` | N/A |
| Copilot CLI Agent | External CLI process | `backend/agents/copilot.go` | N/A |
| Codex CLI Agent | External CLI process | `backend/agents/codex.go` | N/A |

### Dependency Map
```
Frontend (Vite :5173) -> Backend (Gin :3000) [HTTP proxy /api] [fallback: no]
Backend (Gin :3000) -> SQLite (./auction.db) [file-based] [fallback: no]
Backend (Gin :3000) -> claude CLI [os/exec, 20s timeout] [fallback: yes, fallback bid]
Backend (Gin :3000) -> gemini CLI [os/exec, 20s timeout] [fallback: yes, fallback bid]
Backend (Gin :3000) -> copilot CLI [os/exec, 20s timeout] [fallback: yes, fallback bid]
Backend (Gin :3000) -> codex CLI [os/exec, 20s timeout] [fallback: yes, fallback bid]
```

### Startup Order
1. Ensure `auction.db` directory is writable (`backend/`)
2. Start Go backend: `cd backend && go build -o auction-server . && ./auction-server`
3. Start frontend: `cd frontend && bun install && bun run dev`
4. Or use `./run.sh` which does both

### Shutdown Order
1. Stop frontend (Vite dev server on :5173)
2. Stop backend (auction-server on :3000)
3. Or use `./stop.sh` which kills both and cleans PID files

## Health Checks

| Endpoint | Expected Response | Checks |
|---|---|---|
| `GET /api/agents` | 200 JSON array of 4 agent configs | Backend running, router registered |
| `GET /api/auctions` | 200 JSON array (may be empty) | Backend running, SQLite readable |

Manual health check commands:
```bash
curl -s http://localhost:3000/api/agents | head -c 200
curl -s http://localhost:3000/api/auctions | head -c 200
curl -s -o /dev/null -w "%{http_code}" http://localhost:5173/
```

## Failure Runbooks

### Backend Process Crash or Unreachable
**Severity:** Critical
**Blast Radius:** Entire application is unusable — frontend cannot proxy API calls, no auctions can run
**Affected Component:** Go Backend (`backend/main.go:30`)
**Fallback Exists:** No

**Symptoms:**
- Frontend shows network errors on all API calls
- `curl http://localhost:3000/api/agents` returns connection refused
- PID file `/tmp/auction-backend.pid` references a dead process

**Log Breadcrumbs:**
- Check `backend/main.go:16` for `log.Fatal(err)` — SQLite init failure at startup
- Check `backend/main.go:29` for `log.Println("Starting server on :3000")` — confirms startup reached

**Diagnosis Steps:**
```bash
cat /tmp/auction-backend.pid
ps aux | grep auction-server
lsof -i :3000
```

**Resolution Steps:**
```bash
cd /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/agents-auction-hourse
./stop.sh
cd backend && go build -o auction-server . && ./auction-server &
echo $! > /tmp/auction-backend.pid
```

**Rollback Procedure:**
1. Kill any orphan processes: `pkill -f auction-server`
2. Restart via `./run.sh`

**Prevention:**
- Monitor the process with a process supervisor or systemd unit

---

### SQLite Database Lock or Corruption
**Severity:** Critical
**Blast Radius:** All auction creation, reads, and writes fail — backend returns 500 on every data endpoint
**Affected Component:** SQLite (`backend/persistence/db.go:16`, file `backend/auction.db`)
**Fallback Exists:** No

**Symptoms:**
- `POST /api/auctions` returns 500
- `GET /api/auctions` returns 500
- Backend logs show `database is locked` or `disk I/O error`

**Log Breadcrumbs:**
- Check `backend/persistence/db.go:17` for `sql.Open` errors
- Check `backend/persistence/db.go:21` for `DB.Exec` schema creation errors
- Check `backend/engine/auction.go:84` for `persistence.CreateRound` errors logged indirectly

**Diagnosis Steps:**
```bash
ls -la /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/agents-auction-hourse/backend/auction.db*
sqlite3 /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/agents-auction-hourse/backend/auction.db "PRAGMA integrity_check;"
sqlite3 /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/agents-auction-hourse/backend/auction.db "PRAGMA journal_mode;"
```

**Resolution Steps:**
1. If locked, kill all backend processes accessing the DB:
```bash
lsof /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/agents-auction-hourse/backend/auction.db
pkill -f auction-server
```
2. If corrupted, remove WAL/SHM files and let SQLite recover:
```bash
rm -f backend/auction.db-shm backend/auction.db-wal
```
3. If unrecoverable, remove and restart (loses history):
```bash
rm -f backend/auction.db backend/auction.db-shm backend/auction.db-wal
```
4. Restart backend

**Rollback Procedure:**
1. Restore `auction.db` from backup if available

**Prevention:**
- Ensure only one backend process runs at a time
- Monitor disk space on the volume holding `backend/auction.db`

---

### SQLite Disk Full
**Severity:** Critical
**Blast Radius:** All writes fail — auction creation, bid recording, round updates
**Affected Component:** SQLite (`backend/persistence/db.go:16`, WAL mode enabled via `?_journal_mode=WAL`)
**Fallback Exists:** No

**Symptoms:**
- Backend returns 500 on POST/PUT operations
- `auction.db-wal` grows unbounded
- System logs show disk full errors

**Diagnosis Steps:**
```bash
df -h /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/agents-auction-hourse/backend/
du -sh /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/agents-auction-hourse/backend/auction.db*
```

**Resolution Steps:**
1. Free disk space
2. Force a WAL checkpoint to reclaim space:
```bash
sqlite3 /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/agents-auction-hourse/backend/auction.db "PRAGMA wal_checkpoint(TRUNCATE);"
```
3. Optionally vacuum old auctions:
```bash
sqlite3 /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/agents-auction-hourse/backend/auction.db "DELETE FROM bids WHERE auction_id IN (SELECT id FROM auctions WHERE created_at < datetime('now', '-30 days'));"
sqlite3 /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/agents-auction-hourse/backend/auction.db "VACUUM;"
```

**Rollback Procedure:**
1. If data was accidentally deleted, restore from backup

**Prevention:**
- Set up disk usage alerting on the backend directory

---

### Agent CLI Timeout or Failure
**Severity:** Medium
**Blast Radius:** Degraded auction quality — affected agent gets a fallback bid instead of a real one, auction continues
**Affected Component:** Agent Runner (`backend/agents/runner.go:20`, 20-second timeout)
**Fallback Exists:** Yes — `backend/engine/bidding.go:33-40` generates a fallback bid ($5 for first bidder, currentHighest+1 for others)

**Symptoms:**
- Auction proceeds but some agents show `"fallback bid (could not parse agent output)"` as reasoning
- Bid marked with `fallback: true` in the response
- Agent response time near 20000ms (timeout hit)

**Log Breadcrumbs:**
- Check `backend/engine/auction.go:114` for `log.Printf("Agent %s error: %v, output: %s", ...)` — CLI exec failure
- Check `backend/engine/bidding.go:32` for `log.Printf("Failed to parse bid from output: %s", ...)` — parse failure

**Diagnosis Steps:**
```bash
which claude && claude --version
which gemini && gemini --version
which copilot && copilot --version
which codex && codex --version
echo '{"bid": 10, "reasoning": "test"}' | claude -p "respond with only: {\"bid\": 10, \"reasoning\": \"test\"}" --model sonnet --dangerously-skip-permissions
```

**Resolution Steps:**
1. Verify the failing CLI tool is installed and on PATH
2. Verify authentication/API keys are configured for the CLI tool
3. Check if the agent's upstream API is experiencing an outage
4. If a specific agent consistently fails, avoid selecting it in the auction setup

**Rollback Procedure:**
1. No rollback needed — fallback bids are automatically used

**Prevention:**
- Verify all agent CLIs are installed before running auctions
- Use `test.sh` to verify a full auction cycle works

---

### Agent Output Parse Failure
**Severity:** Low
**Blast Radius:** Single bid degraded — agent gets fallback bid, auction continues normally
**Affected Component:** Bid Parser (`backend/engine/bidding.go:15-41`)
**Fallback Exists:** Yes — fallback bid logic at `backend/engine/bidding.go:33-40`

**Symptoms:**
- Agent responds but bid shows `fallback: true`
- Log shows `Failed to parse bid from output: <raw output>`
- Agent response time is normal (not a timeout)

**Log Breadcrumbs:**
- Check `backend/engine/bidding.go:32` for `log.Printf("Failed to parse bid from output: %s", ...)` with the raw output

**Diagnosis Steps:**
1. Check backend logs for the raw output that failed to parse
2. Verify the output contains a JSON object matching `{"bid": <int>, "reasoning": "<string>"}`
3. Regex used: `\{[^{}]*"bid"\s*:\s*\d+[^{}]*\}` (defined at `backend/engine/bidding.go:18`)

**Resolution Steps:**
1. If an agent model consistently returns non-JSON, try a different model for that agent
2. The parser handles bids exceeding budget by capping to budget (`backend/engine/bidding.go:27-28`)

**Rollback Procedure:**
1. No rollback needed

**Prevention:**
- Use models known to follow JSON output instructions reliably

---

### SSE Connection Drop or Channel Overflow
**Severity:** Medium
**Blast Radius:** Individual client loses real-time auction updates — can still poll via `GET /api/auctions/:id`
**Affected Component:** SSE Broadcaster (`backend/sse/broadcaster.go:22`, channel buffer size 100)
**Fallback Exists:** Yes — client can poll `GET /api/auctions/:id` for final state

**Symptoms:**
- Frontend shows "connecting" status indefinitely
- Auction events stop appearing in the live view
- If channel buffer (100) overflows, messages are silently dropped (`backend/sse/broadcaster.go:47-49`, `default:` case)

**Log Breadcrumbs:**
- No explicit logging in the SSE broadcaster — drops are silent
- Check `backend/handlers/stream.go:19` for subscribe path

**Diagnosis Steps:**
```bash
curl -N http://localhost:3000/api/auctions/<AUCTION_ID>/stream
```

**Resolution Steps:**
1. Refresh the browser to establish a new SSE connection
2. If the auction is already finished, navigate to the auction detail page which fetches via REST
3. If connections are consistently dropping, check for reverse proxy or firewall timeout settings

**Rollback Procedure:**
1. No rollback needed — refresh the page

**Prevention:**
- The SSE channel buffer is 100 messages which is sufficient for 3-round auctions (roughly 21 events max)

---

### Frontend Dev Server Unreachable
**Severity:** High
**Blast Radius:** UI unavailable — backend still functions, API is accessible via curl
**Affected Component:** Vite Dev Server (`frontend/vite.config.ts:8`, port 5173)
**Fallback Exists:** No (for UI). Backend API at :3000 still works directly.

**Symptoms:**
- Browser shows connection refused on `http://localhost:5173`
- PID file `/tmp/auction-frontend.pid` references a dead process

**Log Breadcrumbs:**
- Check Vite terminal output for build/compilation errors

**Diagnosis Steps:**
```bash
cat /tmp/auction-frontend.pid
ps aux | grep vite
lsof -i :5173
```

**Resolution Steps:**
```bash
cd /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/agents-auction-hourse/frontend
bun install
bun run dev &
echo $! > /tmp/auction-frontend.pid
```

**Rollback Procedure:**
1. Kill orphan vite processes: `pkill -f "vite.*5173"`
2. Restart via `./run.sh`

**Prevention:**
- Ensure `bun` is installed and `node_modules` are present

---

### Port Conflict on Startup
**Severity:** High
**Blast Radius:** Affected service (backend or frontend) fails to start
**Affected Component:** Backend `:3000` (`backend/main.go:30`), Frontend `:5173` (`frontend/vite.config.ts:8`)
**Fallback Exists:** No

**Symptoms:**
- Backend logs `bind: address already in use` on port 3000
- Vite logs port already in use (may auto-increment to 5174 or 5175 — CORS allows 5173-5175 per `backend/main.go:21`)

**Diagnosis Steps:**
```bash
lsof -i :3000
lsof -i :5173
lsof -i :5174
```

**Resolution Steps:**
1. Kill the process occupying the port:
```bash
lsof -ti:3000 | xargs kill
lsof -ti:5173 | xargs kill
```
2. Or run `./stop.sh` which handles both ports
3. Restart via `./run.sh`

**Rollback Procedure:**
1. `./stop.sh` cleans up all processes and PID files

**Prevention:**
- Always run `./stop.sh` before `./run.sh`

---

### CORS Rejection
**Severity:** Medium
**Blast Radius:** Frontend cannot communicate with backend — all API calls fail in browser
**Affected Component:** CORS config (`backend/main.go:20-25`)
**Fallback Exists:** No

**Symptoms:**
- Browser console shows CORS errors on API requests
- Frontend on a port not in the allowed list (5173, 5174, 5175)
- Backend responds normally via curl but not from browser

**Diagnosis Steps:**
```bash
curl -v -H "Origin: http://localhost:5173" http://localhost:3000/api/agents 2>&1 | grep -i "access-control"
```

**Resolution Steps:**
1. If Vite chose a different port, update `AllowOrigins` in `backend/main.go:21` to include it
2. Rebuild and restart backend
3. Note: Vite proxy at `/api` (`frontend/vite.config.ts:9-13`) should bypass CORS for proxied requests — if CORS errors appear, the proxy may not be working

**Rollback Procedure:**
1. Revert `backend/main.go` changes and restart

**Prevention:**
- The Vite proxy configuration should handle this transparently in dev mode

---

## Configuration Drift Checks
- CORS allowed origins (`backend/main.go:21`) lists ports 5173, 5174, 5175 — matches expected Vite port range
- Vite proxy target (`frontend/vite.config.ts:10`) points to `http://localhost:3000` — matches backend listen port
- Frontend `BASE_URL` (`frontend/src/api/auctions.ts:3`) is empty string — relies on Vite proxy, consistent with `vite.config.ts` proxy setup

## Appendix

### Detected Endpoints
| Method | Path | Handler | File |
|---|---|---|---|
| POST | `/api/auctions` | `handlers.CreateAuction` | `backend/handlers/auction.go:15` |
| GET | `/api/auctions` | `handlers.ListAuctions` | `backend/handlers/history.go:13` |
| GET | `/api/auctions/:id` | `handlers.GetAuction` | `backend/handlers/auction.go:64` |
| GET | `/api/auctions/:id/stream` | `handlers.StreamAuction` | `backend/handlers/stream.go:11` |
| GET | `/api/agents` | `handlers.ListAgents` | `backend/handlers/history.go:25` |

### Detected Dependencies
| Dependency | Type | Connection | Config Location |
|---|---|---|---|
| SQLite | Embedded DB | `./auction.db?_journal_mode=WAL` | `backend/persistence/db.go:16` |
| `claude` CLI | External process | `exec.Command("claude", ...)` | `backend/agents/claude.go:10` |
| `gemini` CLI | External process | `exec.Command("gemini", ...)` | `backend/agents/gemini.go:8` |
| `copilot` CLI | External process | `exec.Command("copilot", ...)` | `backend/agents/copilot.go:10` |
| `codex` CLI | External process | `exec.Command("codex", ...)` | `backend/agents/codex.go:10` |

### Detected Configuration Files
| File | Purpose |
|---|---|
| `backend/go.mod` | Go module dependencies (Go 1.24, Gin, SQLite3, UUID) |
| `frontend/package.json` | Frontend dependencies (React 19, TanStack Router/Query/Table, Tailwind, Vite) |
| `frontend/vite.config.ts` | Vite config with API proxy to backend :3000 |
| `frontend/tailwind.config.js` | Tailwind CSS configuration |
| `frontend/tsconfig.json` | TypeScript configuration |
| `run.sh` | Startup script — builds and runs both services |
| `stop.sh` | Shutdown script — kills both services and cleans PID files |
| `test.sh` | Smoke test — creates an auction via API |

### Detected Resilience Patterns
| Pattern | Location | Details |
|---|---|---|
| Agent CLI timeout | `backend/agents/runner.go:20` | 20-second `context.WithTimeout` on all CLI executions |
| Fallback bid on parse failure | `backend/engine/bidding.go:33-40` | Generates $5 (first bidder) or currentHighest+1 bid if agent output is unparseable |
| Budget cap on overbid | `backend/engine/bidding.go:27-28` | If agent bids more than budget, bid is capped to remaining budget |
| SSE channel buffer | `backend/sse/broadcaster.go:22` | Buffered channel (100) with non-blocking send to prevent slow clients from blocking auction |
| CORS multi-port | `backend/main.go:21` | Allows frontend on ports 5173-5175 to handle Vite port auto-increment |
