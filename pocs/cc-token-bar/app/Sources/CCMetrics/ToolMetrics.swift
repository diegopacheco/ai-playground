import Foundation

public struct ToolLatencyAgg: Codable, Equatable {
    public var count: Int
    public var totalMs: Double
    public init(count: Int = 0, totalMs: Double = 0) {
        self.count = count
        self.totalMs = totalMs
    }
}

public struct ToolEvent {
    public enum Kind {
        case use(id: String, name: String)
        case result(id: String)
    }
    public let kind: Kind
    public let timestamp: Date
    public init(kind: Kind, timestamp: Date) {
        self.kind = kind
        self.timestamp = timestamp
    }
}

public enum ToolMetrics {
    public static func normalizeToolName(_ name: String) -> String {
        if name.hasPrefix("mcp__playwright__") { return "mcp_playwright" }
        return name
    }

    public static func pairLatencies(_ events: [ToolEvent]) -> [String: ToolLatencyAgg] {
        var pending: [String: (name: String, ts: Date)] = [:]
        var out: [String: ToolLatencyAgg] = [:]
        for e in events {
            switch e.kind {
            case let .use(id, name):
                pending[id] = (name, e.timestamp)
            case let .result(id):
                guard let start = pending.removeValue(forKey: id) else { continue }
                let ms = e.timestamp.timeIntervalSince(start.ts) * 1000.0
                guard ms >= 0 else { continue }
                let key = normalizeToolName(start.name)
                var agg = out[key] ?? ToolLatencyAgg()
                agg.count += 1
                agg.totalMs += ms
                out[key] = agg
            }
        }
        return out
    }
}
