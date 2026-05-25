import Foundation
import CCMetrics

struct ModelUsage: Codable {
    var input_tokens: Int
    var output_tokens: Int
    var cache_creation_input_tokens: Int
    var cache_read_input_tokens: Int
    var web_search_requests: Int
    var web_fetch_requests: Int
    var messages: Int
}

struct SessionFile: Codable {
    var session_id: String
    var project_path: String?
    var started_at: String?
    var updated_at: String?
    var by_model: [String: ModelUsage]
    var tool_latency: [String: ToolLatencyAgg]?
}

struct ToolEntry: Codable {
    var count: Int
    var input_bytes: Int
    var output_bytes: Int
}

struct ToolsFile: Codable {
    var session_id: String
    var updated_at: String?
    var tools: [String: ToolEntry]
}

struct PriceTier: Codable {
    var input: Double
    var output: Double
    var cache_write: Double
    var cache_read: Double
}

struct AppConfig: Codable {
    var version: Int
    var store_project_paths: Bool?
    var pricing: [String: PriceTier]
}

struct TokenTotals: Equatable {
    var input: Int = 0
    var output: Int = 0
    var cacheWrite: Int = 0
    var cacheRead: Int = 0
    var costUSD: Double = 0
    var total: Int { input + output + cacheWrite + cacheRead }
}

struct DayBucket: Identifiable, Equatable {
    let id: String
    let label: String
    let input: Int
    let output: Int
}

struct ToolStat: Identifiable, Equatable {
    var id: String { name }
    let name: String
    let count: Int
    let approxTokens: Int
    let costUSD: Double
}

struct ToolLatency: Identifiable, Equatable {
    var id: String { name }
    let name: String
    let count: Int
    let avgMs: Double
    let totalMs: Double
}

struct Aggregates: Equatable {
    var today: TokenTotals = TokenTotals()
    var lifetime: TokenTotals = TokenTotals()
    var byModel: [(String, TokenTotals)] = []
    var byDay: [DayBucket] = []
    var tools: [ToolStat] = []
    var toolLatencies: [ToolLatency] = []
    var cacheHitRatio: Double = 0
    var sessionsToday: Int = 0
    var sessionsLifetime: Int = 0
    var statusLabel: String = ""

    static func == (lhs: Aggregates, rhs: Aggregates) -> Bool {
        lhs.today == rhs.today && lhs.lifetime == rhs.lifetime
            && lhs.byDay == rhs.byDay && lhs.tools == rhs.tools
            && lhs.toolLatencies == rhs.toolLatencies
            && lhs.cacheHitRatio == rhs.cacheHitRatio
            && lhs.sessionsToday == rhs.sessionsToday
            && lhs.sessionsLifetime == rhs.sessionsLifetime
            && lhs.statusLabel == rhs.statusLabel
            && lhs.byModel.map(\.0) == rhs.byModel.map(\.0)
            && lhs.byModel.map(\.1) == rhs.byModel.map(\.1)
    }
}
