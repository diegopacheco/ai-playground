import Foundation
import Combine
import CCMetrics

final class DataStore: ObservableObject {
    @Published private(set) var agg: Aggregates = Aggregates()

    private let dataDir: URL
    private let sessionsDir: URL
    private let toolsDir: URL
    private var watcher: FSWatcher?
    private var visibleTimer: Timer?
    private let queue = DispatchQueue(label: "cc-token-bar.scan", qos: .utility)
    private var pendingRefresh = false
    private let transcripts = TranscriptScanner()

    init() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        self.dataDir = home.appendingPathComponent(".cc-token-bar")
        self.sessionsDir = dataDir.appendingPathComponent("sessions")
        self.toolsDir = dataDir.appendingPathComponent("tools")
    }

    func start() {
        try? FileManager.default.createDirectory(at: sessionsDir, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: toolsDir, withIntermediateDirectories: true)
        watcher = FSWatcher(paths: [sessionsDir.path, toolsDir.path]) { [weak self] in
            self?.scheduleRefresh()
        }
        watcher?.start()
        scheduleRefresh()
    }

    func refreshNow() {
        scheduleRefresh()
    }

    func startVisibleRefresh() {
        stopVisibleRefresh()
        let t = Timer(timeInterval: 5, repeats: true) { [weak self] _ in
            self?.scheduleRefresh()
        }
        RunLoop.main.add(t, forMode: .common)
        visibleTimer = t
    }

    func stopVisibleRefresh() {
        visibleTimer?.invalidate()
        visibleTimer = nil
    }

    private func scheduleRefresh() {
        if pendingRefresh { return }
        pendingRefresh = true
        queue.asyncAfter(deadline: .now() + .milliseconds(250)) { [weak self] in
            self?.pendingRefresh = false
            self?.refresh()
        }
    }

    private func refresh() {
        let cfg = Pricing.loadConfig(from: dataDir)
        let pricing = cfg.pricing.isEmpty ? Pricing.fallback : cfg.pricing
        let sessions = mergedSessions()
        let tools = loadTools()
        let next = aggregate(sessions: sessions, tools: tools, pricing: pricing)
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            if self.agg != next { self.agg = next }
        }
    }

    private func mergedSessions() -> [SessionFile] {
        let live = transcripts.scan()
        let cached = loadSessions()
        var merged: [String: SessionFile] = [:]
        for s in cached { merged[s.session_id] = s }
        for s in live { merged[s.session_id] = s }
        return Array(merged.values)
    }

    private func loadSessions() -> [SessionFile] {
        let fm = FileManager.default
        guard let urls = try? fm.contentsOfDirectory(at: sessionsDir, includingPropertiesForKeys: nil) else {
            return []
        }
        let dec = JSONDecoder()
        var out: [SessionFile] = []
        out.reserveCapacity(urls.count)
        for url in urls where url.pathExtension == "json" {
            if let data = try? Data(contentsOf: url),
               let s = try? dec.decode(SessionFile.self, from: data) {
                out.append(s)
            }
        }
        return out
    }

    private func loadTools() -> [ToolsFile] {
        let fm = FileManager.default
        guard let urls = try? fm.contentsOfDirectory(at: toolsDir, includingPropertiesForKeys: nil) else {
            return []
        }
        let dec = JSONDecoder()
        var out: [ToolsFile] = []
        out.reserveCapacity(urls.count)
        for url in urls where url.pathExtension == "json" {
            if let data = try? Data(contentsOf: url),
               let t = try? dec.decode(ToolsFile.self, from: data) {
                out.append(t)
            }
        }
        return out
    }

    private static let isoFrac: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()
    private static let isoBasic = ISO8601DateFormatter()

    private static func parseISO(_ s: String) -> Date? {
        if let d = isoFrac.date(from: s) { return d }
        return isoBasic.date(from: s)
    }

    private func aggregate(sessions: [SessionFile], tools: [ToolsFile], pricing: [String: PriceTier]) -> Aggregates {
        let cal = Calendar(identifier: .gregorian)
        let todayKey = Self.dayKey(for: Date(), cal: cal)
        let oneDay: TimeInterval = 86_400
        let sevenDaysAgo = Date().addingTimeInterval(-6 * oneDay)

        var lifetime = TokenTotals()
        var today = TokenTotals()
        var byModelMap: [String: TokenTotals] = [:]
        var byDayMap: [String: (input: Int, output: Int)] = [:]
        var sessionsTodaySet: Set<String> = []

        for s in sessions {
            let when = s.updated_at.flatMap { Self.parseISO($0) }
                ?? s.started_at.flatMap { Self.parseISO($0) }
            let dayKey = when.map { Self.dayKey(for: $0, cal: cal) } ?? todayKey
            let isToday = dayKey == todayKey
            let inWeek = when.map { $0 >= sevenDaysAgo } ?? false

            for (model, usage) in s.by_model {
                if Self.isSyntheticModel(model) { continue }
                let tier = Pricing.tier(for: model, table: pricing)
                let cost = Pricing.cost(usage, tier: tier)
                add(&lifetime, usage: usage, cost: cost)
                var m = byModelMap[model] ?? TokenTotals()
                add(&m, usage: usage, cost: cost)
                byModelMap[model] = m
                if isToday {
                    add(&today, usage: usage, cost: cost)
                    sessionsTodaySet.insert(s.session_id)
                }
                if inWeek {
                    var d = byDayMap[dayKey] ?? (0, 0)
                    d.input  += usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
                    d.output += usage.output_tokens
                    byDayMap[dayKey] = d
                }
            }
        }

        let weekKeys: [String] = (0..<7).map { i in
            let d = Date().addingTimeInterval(-Double(6 - i) * oneDay)
            return Self.dayKey(for: d, cal: cal)
        }
        let weekLabels: [String] = (0..<7).map { i in
            let d = Date().addingTimeInterval(-Double(6 - i) * oneDay)
            return Self.shortDayLabel(for: d, cal: cal)
        }
        let byDay: [DayBucket] = zip(weekKeys, weekLabels).map { (key, label) in
            let v = byDayMap[key] ?? (0, 0)
            return DayBucket(id: key, label: label, input: v.input, output: v.output)
        }

        let avgInputPricePerM: Double = {
            var totalInput = 0
            var totalCost = 0.0
            for (model, t) in byModelMap {
                let tier = Pricing.tier(for: model, table: pricing)
                totalInput += t.input
                totalCost += Double(t.input) * tier.input / 1_000_000.0
            }
            guard totalInput > 0 else { return 5.0 }
            return totalCost / Double(totalInput) * 1_000_000.0
        }()

        var toolStats: [ToolStat] = []
        var toolAcc: [String: (count: Int, bytes: Int)] = [:]
        for tf in tools {
            for (name, e) in tf.tools {
                let key = ToolMetrics.normalizeToolName(name)
                var v = toolAcc[key] ?? (0, 0)
                v.count += e.count
                v.bytes += e.input_bytes + e.output_bytes
                toolAcc[key] = v
            }
        }
        for (name, v) in toolAcc {
            let approxTokens = v.bytes / 4
            let cost = Double(approxTokens) / 1_000_000.0 * avgInputPricePerM
            toolStats.append(ToolStat(name: name, count: v.count, approxTokens: approxTokens, costUSD: cost))
        }
        toolStats.sort { $0.costUSD > $1.costUSD }

        var latAcc: [String: (count: Int, total: Double)] = [:]
        for s in sessions {
            guard let tl = s.tool_latency else { continue }
            for (name, agg) in tl {
                let key = ToolMetrics.normalizeToolName(name)
                var v = latAcc[key] ?? (0, 0)
                v.count += agg.count
                v.total += agg.totalMs
                latAcc[key] = v
            }
        }
        var toolLatencies: [ToolLatency] = latAcc.map { (name, v) in
            ToolLatency(name: name, count: v.count,
                        avgMs: v.count > 0 ? v.total / Double(v.count) : 0,
                        totalMs: v.total)
        }
        toolLatencies.sort { $0.avgMs > $1.avgMs }

        let cacheReads = byModelMap.values.reduce(0) { $0 + $1.cacheRead }
        let cacheDen = byModelMap.values.reduce(0) { $0 + $1.input + $1.cacheWrite + $1.cacheRead }
        let cacheRatio = cacheDen > 0 ? Double(cacheReads) / Double(cacheDen) : 0

        let byModelSorted = byModelMap
            .map { ($0.key, $0.value) }
            .sorted { $0.1.costUSD > $1.1.costUSD }

        let label = Self.formatStatusLabel(today: today)

        return Aggregates(
            today: today,
            lifetime: lifetime,
            byModel: byModelSorted,
            byDay: byDay,
            tools: Array(toolStats.prefix(10)),
            toolLatencies: Array(toolLatencies.prefix(10)),
            cacheHitRatio: cacheRatio,
            sessionsToday: sessionsTodaySet.count,
            sessionsLifetime: sessions.count,
            statusLabel: label
        )
    }

    private func add(_ t: inout TokenTotals, usage: ModelUsage, cost: Double) {
        t.input      += usage.input_tokens
        t.output     += usage.output_tokens
        t.cacheWrite += usage.cache_creation_input_tokens
        t.cacheRead  += usage.cache_read_input_tokens
        t.costUSD    += cost
    }

    static func dayKey(for date: Date, cal: Calendar) -> String {
        let c = cal.dateComponents([.year, .month, .day], from: date)
        return String(format: "%04d-%02d-%02d", c.year ?? 0, c.month ?? 0, c.day ?? 0)
    }

    static func shortDayLabel(for date: Date, cal: Calendar) -> String {
        let f = DateFormatter()
        f.calendar = cal
        f.dateFormat = "EEE"
        return f.string(from: date)
    }

    static func formatStatusLabel(today: TokenTotals) -> String {
        return "\(formatTokens(today.total)) \(formatUSD(today.costUSD))"
    }

    private static let groupedIntFormatter: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.groupingSeparator = ","
        f.maximumFractionDigits = 0
        return f
    }()

    private static let usdFormatter: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .currency
        f.currencyCode = "USD"
        f.currencySymbol = "$"
        f.minimumFractionDigits = 2
        f.maximumFractionDigits = 2
        return f
    }()

    static func formatTokens(_ n: Int) -> String {
        let d = Double(n)
        if d >= 1_000_000_000 { return String(format: "%.2fB", d / 1_000_000_000) }
        if d >= 1_000_000     { return String(format: "%.1fM", d / 1_000_000) }
        if d >= 10_000        { return String(format: "%.0fk", d / 1_000) }
        if d >= 1_000         { return String(format: "%.1fk", d / 1_000) }
        return groupedIntFormatter.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    static func formatUSD(_ v: Double) -> String {
        if v >= 1_000_000 { return String(format: "$%.2fM", v / 1_000_000) }
        return usdFormatter.string(from: NSNumber(value: v)) ?? String(format: "$%.2f", v)
    }

    static func formatCount(_ n: Int) -> String {
        return groupedIntFormatter.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    static func formatMs(_ ms: Double) -> String {
        if ms >= 1000 { return String(format: "%.2fs", ms / 1000) }
        return String(format: "%.0f ms", ms)
    }

    static func isSyntheticModel(_ name: String) -> Bool {
        return name.hasPrefix("<") || name.lowercased().contains("synthetic")
    }
}
