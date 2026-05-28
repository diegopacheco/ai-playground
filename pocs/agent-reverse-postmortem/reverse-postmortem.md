# Reverse Postmortem — orders (sample)

Generated: 2026-05-27
Incidents predicted: 5   |   P0: 2  P1: 2  P2: 1  P3: 0

## How to read this
These incidents have NOT happened. Each is the postmortem of a likely future
outage, written in advance so the cause can be fixed before it fires. Ordered
worst-first by Likelihood x Blast Radius.

## Risk Summary

| # | Incident | Likelihood | Blast Radius | Risk | Tier |
|---|----------|-----------|--------------|------|------|
| INC-1 | Checkout thread hang from timeout-less payment calls | 5 | 5 | 25 | P0 |
| INC-2 | Connection pool exhaustion under load (N+1) | 5 | 4 | 20 | P0 |
| INC-3 | Worker death on poison event stalls checkout | 4 | 4 | 16 | P1 |
| INC-4 | Unbounded totals cache OOM / concurrent-map crash | 4 | 3 | 12 | P1 |
| INC-5 | Single replica is a total outage on any crash | 2 | 4 | 8 | P2 |

---

## INC-1: Checkout thread hang from timeout-less payment calls

**Tier:** P0
**Risk Score:** 5 x 5 = 25
**Predicted trigger:** payments.internal slows down or stops responding (deploy, GC pause, network blip)
**Affected components:** `/orders/checkout` handler, payments dependency

### Summary
The payment provider got slow for 6 minutes. Because every charge used the default
HTTP client with no timeout and retried five times, each checkout request held its
goroutine open indefinitely and multiplied load 5x against an already-struggling
dependency. Checkout returned errors or hung for ~14 minutes and the provider's
recovery was delayed by our own retry storm.

### Timeline (predicted)
- T+0:00  payments.internal latency rises from 80ms to 30s+ (provider-side GC pause)
- T+0:30  `http.Post` calls in `charge` block with no deadline; checkout goroutines pile up
- T+1:00  each stuck request retries up to 5x with no backoff, multiplying load on the provider
- T+2:00  no alert fires; discovered via customer reports of "checkout spinning"
- T+8:00  provider recovers but our retry storm keeps it saturated
- T+14:00 traffic drains, checkout normalizes
- T+20:00 manual: payment client timeout added and deployed

### Root Cause
`http.Post` uses `http.DefaultClient`, which has no timeout, and the retry loop has
no backoff and no circuit breaker.
**Evidence:** `sample/payments.go:15`
```go
	for attempt := 0; attempt < 5; attempt++ {
		resp, err := http.Post(paymentsURL, "application/json", bytes.NewReader(body))
		if err != nil {
			continue
		}
```
A timeout-less call can block forever; the tight retry loop turns one slow dependency
into 5x self-inflicted load that prevents the dependency from recovering.

### Contributing Factors
- No `http.Client{Timeout: ...}` anywhere in the codebase
- Retry has no exponential backoff or jitter (`sample/payments.go:14`)
- No circuit breaker / fallback around the payment dependency

### Detection Gap
- Existing signals near this path: none — no metrics, no latency histogram, no log on retry
- What's missing: a payment-latency metric + alert, and a request-level timeout that would surface failures fast instead of hanging

### Blast Radius Detail
`/orders/checkout` is the revenue path. With goroutines blocked, the default server
accepts connections but never responds; the impact is a full checkout outage plus
prolonged damage to the shared payment provider used by other services.

### Action Items (prevent this incident)
| # | Action | File / Area | Effort | Prevents |
|---|--------|-------------|--------|----------|
| 1 | Use `http.Client{Timeout: 3s}` instead of `http.Post` | `sample/payments.go:15` | S | the indefinite hang |
| 2 | Add exponential backoff + jitter, cap attempts | `sample/payments.go:14` | S | the retry storm |
| 3 | Wrap charge in a circuit breaker with a fast-fail fallback | `sample/payments.go:12` | M | prolonged provider damage |
| 4 | Emit a payment latency/error metric and alert | `sample/payments.go` | M | the detection gap |

### Earliest Intervention Point
Set a 3-second client timeout on the payment call. One line removes the indefinite
hang and turns a silent outage into a fast, observable failure.

---

## INC-2: Connection pool exhaustion under load (N+1)

**Tier:** P0
**Risk Score:** 5 x 4 = 20
**Predicted trigger:** a modest traffic increase on `/orders` for users with many orders
**Affected components:** `/orders`, `/orders/checkout`, Postgres

### Summary
A marketing email drove traffic to order history. Each `/orders` call ran one query
per order (N+1), and with the pool capped at 5 connections, the pool drained in
seconds. Every endpoint that touches the database — including checkout — started
timing out. The outage lasted 11 minutes until traffic subsided.

### Timeline (predicted)
- T+0:00  `/orders` traffic doubles; power users have 50+ orders each
- T+0:10  each request issues 1 + N queries via `fetchItems` in a loop
- T+0:20  the 5-connection pool is fully checked out; new queries block
- T+0:40  checkout (`cachedTotal` -> `computeTotal` -> `fetchOrdersWithItems`) also blocks
- T+1:00  no alert; discovered when checkout error rate is noticed manually
- T+11:00 traffic subsides, pool recovers

### Root Cause
`fetchOrdersWithItems` calls `fetchItems` once per order inside the row loop (N+1),
and the pool is capped at 5 with no per-query timeout.
**Evidence:** `sample/db.go:41`
```go
	for rows.Next() {
		var o Order
		rows.Scan(&o.ID, &o.UserID, &o.Total)
		o.Items = fetchItems(o.ID)
		orders = append(orders, o)
	}
```
Combined with `db.SetMaxOpenConns(5)` (`sample/db.go:30`), a handful of heavy users
saturate the pool and block the entire service, including the revenue path.

### Contributing Factors
- `MaxOpenConns(5)` is very low for a shared service (`sample/db.go:30`, `config.yaml:9`)
- No `context` deadline on `db.Query` — blocked checkouts never time out
- Query errors are ignored (`rows, _ := db.Query(...)`), hiding the failure

### Detection Gap
- Existing signals near this path: none — no pool-usage metric, no slow-query log
- What's missing: a `db.Stats().InUse` gauge with an alert near the max, and query timeouts

### Blast Radius Detail
The pool is shared across all DB-backed endpoints, so exhaustion from `/orders`
read traffic takes down `/orders/checkout` writes too. Degrades to a full DB-path outage.

### Action Items (prevent this incident)
| # | Action | File / Area | Effort | Prevents |
|---|--------|-------------|--------|----------|
| 1 | Replace N+1 with a single JOIN or `WHERE order_id = ANY($1)` batch | `sample/db.go:41` | M | pool drain at the source |
| 2 | Raise `MaxOpenConns` to a load-tested value | `sample/db.go:30` | S | premature saturation |
| 3 | Use `QueryContext` with a deadline on all queries | `sample/db.go:37,49` | M | blocked-forever checkouts |
| 4 | Stop ignoring query errors; log and surface them | `sample/db.go:37` | S | silent failures |

### Earliest Intervention Point
Collapse the N+1 into a single query. It removes the multiplier that drains the pool,
without touching connection limits.

---

## INC-3: Worker death on poison event stalls checkout

**Tier:** P1
**Risk Score:** 4 x 4 = 16
**Predicted trigger:** any event enqueued with a type other than "order.paid"
**Affected components:** background worker, checkout (via the enqueue channel)

### Summary
A new code path enqueued an event type the worker didn't recognize. The worker
`panic`-ed, its single goroutine died, and the unbuffered-after-full channel filled
to 1000. Once full, `enqueue` blocked inside the checkout handler, so checkout hung
even though payments were healthy.

### Timeline (predicted)
- T+0:00  an `order.refunded` (or malformed) event is enqueued
- T+0:01  `process` hits the `default` branch and panics; the only worker goroutine exits
- T+5:00  the buffered channel (cap 1000) fills as checkouts keep enqueuing `order.paid`
- T+5:30  `enqueue(e)` blocks; `/orders/checkout` hangs after a successful charge
- T+6:00  no alert; discovered via checkout latency
- T+15:00 manual restart clears it, but the poison event recurs on next deploy

### Root Cause
The worker panics on unknown event types, runs as a single goroutine with no recover,
and there is no dead-letter handling or max-retry.
**Evidence:** `sample/worker.go:29`
```go
	default:
		panic("unknown event type: " + e.Type)
```
One unexpected message kills the consumer permanently; the bounded channel then
back-pressures into the synchronous checkout path.

### Contributing Factors
- Single consumer goroutine, no `recover()` (`sample/worker.go:16`)
- No dead-letter queue or max-retry cap
- `enqueue` writes to a bounded channel from the request path (`sample/worker.go:12`), coupling the worker's health to checkout latency

### Detection Gap
- Existing signals near this path: a single `log.Printf` on success only
- What's missing: a panic/restart counter and a queue-depth gauge with an alert

### Blast Radius Detail
Worker death alone delays side effects (cache invalidation). But because `enqueue`
is called synchronously in checkout and the channel is bounded, a dead worker
escalates into a checkout hang once the buffer fills.

### Action Items (prevent this incident)
| # | Action | File / Area | Effort | Prevents |
|---|--------|-------------|--------|----------|
| 1 | Log-and-skip unknown event types instead of panicking | `sample/worker.go:29` | S | worker death |
| 2 | Wrap `process` in `recover()` and keep the loop alive | `sample/worker.go:18` | S | one bad message killing the consumer |
| 3 | Make `enqueue` non-blocking (select/default) or decouple from the request path | `sample/worker.go:12` | M | checkout hang on back-pressure |
| 4 | Add a queue-depth metric + alert | `sample/worker.go` | M | the detection gap |

### Earliest Intervention Point
Replace the `panic` with a logged skip. The worker survives unknown messages and the
back-pressure cascade never starts.

---

## INC-4: Unbounded totals cache OOM / concurrent-map crash

**Tier:** P1
**Risk Score:** 4 x 3 = 12
**Predicted trigger:** steady traffic across many distinct users over days; or concurrent checkout requests
**Affected components:** in-memory totals cache, whole process

### Summary
`totalsCache` grew one entry per unique user forever, and was read/written from
concurrent HTTP goroutines and the worker with no lock. Over four days memory
climbed until the process was OOM-killed; separately, a concurrent map write
triggered Go's fatal "concurrent map writes" crash under a traffic burst.

### Timeline (predicted)
- T+0:00  cache adds an entry per new user in `cachedTotal`, never evicting
- T+2d    resident memory grows steadily; no cap, no TTL
- T+4d    process OOM-killed and restarts, briefly dropping requests
- (alt)   under concurrent checkout + worker `delete`, Go aborts with "fatal error: concurrent map writes"

### Root Cause
`totalsCache` is a plain map used as an unbounded, unsynchronized cache. It is
written in `cachedTotal` and deleted in the worker with no mutex.
**Evidence:** `sample/cache.go:3`
```go
var totalsCache = map[string]float64{}

func cachedTotal(userID string) float64 {
	if v, ok := totalsCache[userID]; ok {
		return v
	}
	total := computeTotal(userID)
	totalsCache[userID] = total
	return total
}
```
No eviction means monotonic growth; no lock means concurrent access from request
goroutines and the worker can crash the process.

### Contributing Factors
- No size cap, no TTL, no LRU eviction
- Concurrent writers: HTTP handlers and `process`/`delete` in `sample/worker.go:26`
- No memory metric to see the growth

### Detection Gap
- Existing signals near this path: none
- What's missing: a process-memory alert and a cache-size gauge

### Blast Radius Detail
A crash or OOM restarts the whole process, dropping in-flight requests across all
endpoints. The concurrent-map crash is abrupt and unrecoverable without a fix.

### Action Items (prevent this incident)
| # | Action | File / Area | Effort | Prevents |
|---|--------|-------------|--------|----------|
| 1 | Replace with a bounded LRU + TTL cache | `sample/cache.go:3` | M | unbounded growth / OOM |
| 2 | Guard with `sync.RWMutex` or use `sync.Map` | `sample/cache.go:3` | S | the concurrent-map crash |
| 3 | Emit a memory + cache-size metric | `sample/cache.go` | M | the detection gap |

### Earliest Intervention Point
Add a mutex around the map. It removes the immediate crash risk; bounded eviction
can follow to address the slower OOM.

---

## INC-5: Single replica is a total outage on any crash

**Tier:** P2
**Risk Score:** 2 x 4 = 8
**Predicted trigger:** any process crash, deploy, or node failure
**Affected components:** the whole service

### Summary
The service runs as a single replica. Any crash from the incidents above — or a
routine deploy — produced a full outage with no instances left to serve traffic.

### Timeline (predicted)
- T+0:00  the single instance crashes (OOM from INC-4, or a deploy restart)
- T+0:01  100% of traffic fails — there is no second replica
- T+0:30  instance restarts; cold caches cause a latency spike
- T+1:00  service recovers

### Root Cause
The deployment is configured for one replica.
**Evidence:** `sample/config.yaml:3`
```yaml
  replicas: 1
```
With a single instance there is no redundancy; every restart is a user-visible outage.

### Contributing Factors
- No readiness gating to shift traffic during restarts
- Amplifies every other incident's blast radius

### Detection Gap
- Existing signals near this path: none
- What's missing: an availability/uptime probe and alert

### Blast Radius Detail
Total outage during any restart. Also makes INC-1 through INC-4 worse, since each
crash takes the only instance down.

### Action Items (prevent this incident)
| # | Action | File / Area | Effort | Prevents |
|---|--------|-------------|--------|----------|
| 1 | Run >= 2 replicas behind the load balancer | `sample/config.yaml:3` | S | restart = outage |
| 2 | Add readiness checks so traffic drains before restart | deploy config | M | cold-restart errors |

### Earliest Intervention Point
Set `replicas: 2`. Redundancy alone converts most single-instance crashes from
outages into non-events.

---

## Systemic Patterns
- **Missing timeouts and guards on every external/IO call.** INC-1 (payments) and
  INC-2 (DB) both stem from calls that can block forever with no deadline. A shared
  HTTP client with a sane timeout plus `QueryContext` everywhere removes both root causes.
- **Total absence of observability.** Every incident has the same detection gap:
  no metrics, no alerts, only a single success log. The team would learn of every one
  of these from customers, not dashboards. A minimal metrics layer (latency, pool
  usage, queue depth, memory) is the highest-leverage cross-cutting fix.
- **Synchronous coupling of the request path to fragile subsystems.** Checkout is
  coupled to payments (INC-1), the DB pool (INC-2), and the worker channel (INC-3),
  so any one of them takes checkout down.

## Appendix
### Dependency Map (detected)
```
/orders          -> Postgres (read, N+1)
/orders/create   -> Postgres (write)
/orders/checkout -> totalsCache -> Postgres (read)
                 -> payments.internal (HTTP, no timeout)
                 -> queue channel -> worker -> totalsCache (delete)
worker           -> totalsCache (delete), panics on unknown event
```
Fallbacks detected: none. Circuit breakers: none. Timeouts: none.

### Files Scanned
`sample/main.go`, `sample/db.go`, `sample/cache.go`, `sample/payments.go`, `sample/worker.go`, `sample/config.yaml`, `sample/go.mod`

### Scan Methodology
Static read-only scan. Components and fragility signals detected via pattern grep
and code reading; likelihood weighted by hot-path placement and absence of guards;
blast radius weighted by shared-resource coupling. Timelines are predicted, not historical.
