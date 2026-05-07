# Trailblaze

https://github.com/block/trailblaze

```
⏺ Trailblaze is an AI-powered UI testing framework for iOS, Android, and Web. Based on the README, here's what it covers from your list:

Yes — directly supported:
- E2E: Core use case. Drives real devices/browsers (Android UiAutomator, iOS XCUITest, Playwright DOM) end-to-end via natural-language goals or
scripted .trail.yaml flows.
- UI / functional: The whole product — blaze, verify, ask, snapshot, tapOnElement, etc.
- Visual / vision-based assertions: verify and ask use AI vision to pass/fail conditions on screen.
- Replay / regression: Deterministic replay of recorded trails with --self-heal for UI drift.

Not directly supported (out of scope for this tool):
- Integration tests (service/API layer) — no, it's UI-only.
- Contract tests (Pact-style) — no.
- Snapshot tests (Jest-style serialized output diffing) — no. It records sessions/trails, but that's replay, not snapshot assertions.
- Property-based tests (QuickCheck-style) — no.
- Chaos tests — no.
- Stress / load tests — no.
- CSS tests (visual regression / pixel diffing of styles) — no dedicated CSS or pixel-diff testing; it asserts via AI vision and DOM/accessibility
trees, not CSS rules or screenshot diffs.

Short version: Trailblaze is an E2E UI testing framework with AI-driven authoring and self-healing replay. For integration, contract, snapshot,
property-based, chaos, stress, or CSS testing you'd reach for other tools (RestAssured/Pact/Jest/jqwik/Chaos Monkey/k6/Percy, etc.).
```

## Result

```
❯ ./start.sh
Server up at http://localhost:8080/ (pid 43925)
❯ ./test.sh
Server already running on port 8080 (pid 43925)
Starting Trailblaze daemon...
Waiting for Trailblaze daemon..
Trailblaze daemon ready.
Shutting down existing daemon on port 52525...
Running 1 trail file(s)
============================================================
WARNING: A restricted method in java.lang.System has been called
WARNING: java.lang.System::loadLibrary has been called by io.netty.util.internal.NativeLibraryUtil in an unnamed module (file:/Users/diegopacheco/.trailblaze/bin/trailblaze.jar)
WARNING: Use --enable-native-access=ALL-UNNAMED to avoid a warning for callers in this module
WARNING: Restricted methods will be blocked in a future release unless native access is enabled


[1/1] Running: web.trail.yaml
------------------------------------------------------------
Detected web trail — using Playwright browser
Auto-selected device for platform 'WEB': playwright-native
Target device: playwright-native (Web Browser)
Driver: PLAYWRIGHT_NATIVE
Using LLM: ollama/gpt-oss:20b
Agent: TRAILBLAZE_RUNNER

Starting trail execution...
============================================================
[playwright-native] Starting Web Browser test on device playwright-native with driver type PLAYWRIGHT_NATIVE
kotlin-logging: initializing... active logger factory: Slf4jLoggerFactory
[playwright-native] Initializing Playwright-native test runner...
[playwright-native] Launching browser...
[playwright-native] Executing YAML test...
[playwright-native] Step 1/9: Navigate to the local sample app
[playwright-native] Step 2/9: Click the Counter navigation link
[playwright-native] Step 3/9: Verify the counter starts at 0
[playwright-native] Step 4/9: Click Increment three times
[playwright-native] Step 5/9: Verify the counter shows 3
[playwright-native] Step 6/9: Click Decrement once
[playwright-native] Step 7/9: Verify the counter shows 2
[playwright-native] Step 8/9: Click Reset
[playwright-native] Step 9/9: Verify the counter is back at 0
[playwright-native] Test execution completed successfully
Trace posted to server for session web_trail_70d22194

✅ Trail completed successfully!

============================================================
Results: 1 passed, 0 failed out of 1 total


Execution stopped by user.
Stopped server pid 43925
TRAIL PASSED
```