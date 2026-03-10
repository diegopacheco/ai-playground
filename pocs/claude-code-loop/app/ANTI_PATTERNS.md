# Anti-Patterns and Bad Choices Log

## 1. Global Mutable State
The app uses `var globalStocks []Stock` as a package-level global variable shared across goroutines. This makes the code hard to test, hard to reason about, and tightly couples every handler to a single shared slice.

**Better approach:** Pass dependencies through structs. Use a `Server` struct that holds the stock data and has methods as handlers.

## 2. Mutex Instead of Channels
Uses `sync.Mutex` to coordinate access between the price simulator goroutine and the HTTP handlers. Go idiom says "share memory by communicating" not "communicate by sharing memory."

**Better approach:** Use channels or `sync.RWMutex` at minimum (readers don't block each other).

## 3. Inline HTML in Go Code
The entire HTML/CSS/JS dashboard is embedded as a raw string inside `dashboardHandler`. This makes it impossible to edit the frontend without recompiling, mixes concerns, and no syntax highlighting.

**Better approach:** Use `embed.FS` to embed static files or serve from a `static/` directory with `http.FileServer`.

## 4. No Error Handling on json.Marshal
The line `data, _ := json.Marshal(globalStocks)` silently discards the error. If marshaling fails the client gets an empty response with a 200 status.

**Better approach:** Check the error and return a 500 status with an error message.

## 5. No Error Handling on ListenAndServe
`http.ListenAndServe(":8080", nil)` returns an error that is never checked. If the port is busy the app silently does nothing.

**Better approach:** `log.Fatal(http.ListenAndServe(":8080", nil))`.

## 6. No Graceful Shutdown
The server cannot be stopped cleanly. The `simulatePrices` goroutine runs in an infinite loop with no way to cancel it. No signal handling, no context cancellation.

**Better approach:** Use `context.Context`, `os.Signal`, and `http.Server.Shutdown()`.

## 7. Hardcoded Port
Port 8080 is hardcoded. No way to configure it via environment variable or flag.

**Better approach:** Use `os.Getenv("PORT")` or `flag.String`.

## 8. Using Default ServeMux
`http.HandleFunc` registers on the default global mux. Multiple packages registering routes can collide silently. No middleware support, no route parameters.

**Better approach:** Create an explicit `http.NewServeMux()` and pass it to `ListenAndServe`.

## 9. No CORS Headers
The `/api/stocks` endpoint has no CORS headers. If someone tries to call it from a different origin it will fail.

**Better approach:** Add `Access-Control-Allow-Origin` header or use middleware.

## 10. Fake Data with Biased Random Walk
The price simulation uses `(rand.Float64() - 0.48) * 5.0` which biases prices upward over time (center is 0.52 not 0.50). This is a subtle numeric bug.

**Better approach:** Use `(rand.Float64() - 0.50)` for an unbiased random walk.

## 11. No Logging
Zero logging anywhere. No request logging, no error logging, no startup diagnostics beyond a single println.

**Better approach:** Use `log` or `slog` package for structured logging.

## 12. Polling Instead of WebSockets
The frontend polls every 2 seconds with XHR. This wastes bandwidth and adds latency.

**Better approach:** Use WebSockets or Server-Sent Events (SSE) for real-time push updates.

## 13. XMLHttpRequest Instead of Fetch
The frontend uses the old `XMLHttpRequest` API instead of the modern `fetch` API.

**Better approach:** Use `fetch("/api/stocks").then(r => r.json()).then(render)`.

## 14. innerHTML for Rendering
Uses `innerHTML` to build DOM, which is slow and can be an XSS vector if data is not sanitized.

**Better approach:** Use `textContent` for text nodes and `createElement` for structure, or use a lightweight template approach.

## 15. No Tests
Zero test files. No unit tests, no integration tests, no HTTP handler tests.

**Better approach:** Write `main_test.go` with table-driven tests for handlers and price simulation.
