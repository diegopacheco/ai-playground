import Foundation
import CCMetrics

var failures = 0

func check(_ name: String, _ condition: Bool) {
    if condition {
        print("ok   - \(name)")
    } else {
        print("FAIL - \(name)")
        failures += 1
    }
}

func date(_ s: String) -> Date {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return f.date(from: s)!
}

check("playwright subtools collapse to mcp_playwright",
      ToolMetrics.normalizeToolName("mcp__playwright__browser_click") == "mcp_playwright"
      && ToolMetrics.normalizeToolName("mcp__playwright__browser_navigate") == "mcp_playwright"
      && ToolMetrics.normalizeToolName("mcp__playwright__browser_snapshot") == "mcp_playwright")

check("non-playwright tools keep their name",
      ToolMetrics.normalizeToolName("Read") == "Read"
      && ToolMetrics.normalizeToolName("Bash") == "Bash"
      && ToolMetrics.normalizeToolName("mcp__repo-mcp__grep") == "mcp__repo-mcp__grep")

do {
    let events: [ToolEvent] = [
        ToolEvent(kind: .use(id: "a", name: "Bash"), timestamp: date("2026-05-25T18:48:49.000Z")),
        ToolEvent(kind: .result(id: "a"), timestamp: date("2026-05-25T18:48:51.000Z")),
    ]
    let out = ToolMetrics.pairLatencies(events)
    check("latency = result_ts - use_ts (2000ms)",
          out["Bash"]?.count == 1 && abs((out["Bash"]?.totalMs ?? 0) - 2000) < 0.5)
}

do {
    let events: [ToolEvent] = [
        ToolEvent(kind: .use(id: "a", name: "mcp__playwright__browser_click"), timestamp: date("2026-05-25T18:00:00.000Z")),
        ToolEvent(kind: .result(id: "a"), timestamp: date("2026-05-25T18:00:01.000Z")),
        ToolEvent(kind: .use(id: "b", name: "mcp__playwright__browser_navigate"), timestamp: date("2026-05-25T18:00:02.000Z")),
        ToolEvent(kind: .result(id: "b"), timestamp: date("2026-05-25T18:00:05.000Z")),
    ]
    let out = ToolMetrics.pairLatencies(events)
    check("playwright latency aggregates across subtools (1 row, 2 calls, 4000ms)",
          out.count == 1 && out["mcp_playwright"]?.count == 2 && abs((out["mcp_playwright"]?.totalMs ?? 0) - 4000) < 0.5)
}

check("result without matching use is ignored",
      ToolMetrics.pairLatencies([ToolEvent(kind: .result(id: "orphan"), timestamp: date("2026-05-25T18:00:00.000Z"))]).isEmpty)

do {
    let events: [ToolEvent] = [
        ToolEvent(kind: .use(id: "a", name: "Read"), timestamp: date("2026-05-25T18:00:05.000Z")),
        ToolEvent(kind: .result(id: "a"), timestamp: date("2026-05-25T18:00:00.000Z")),
    ]
    check("negative delta is discarded", ToolMetrics.pairLatencies(events).isEmpty)
}

if failures == 0 {
    print("\nALL PASS")
    exit(0)
} else {
    print("\n\(failures) FAILED")
    exit(1)
}
