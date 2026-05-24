import Foundation

enum Pricing {
    static let fallback: [String: PriceTier] = [
        "claude-opus-4":   PriceTier(input: 15.0, output: 75.0, cache_write: 18.75, cache_read: 1.50),
        "claude-sonnet-4": PriceTier(input:  3.0, output: 15.0, cache_write:  3.75, cache_read: 0.30),
        "claude-haiku-4":  PriceTier(input:  1.0, output:  5.0, cache_write:  1.25, cache_read: 0.10),
    ]

    static func tier(for model: String, table: [String: PriceTier]) -> PriceTier {
        let lower = model.lowercased()
        if let exact = table[lower] { return exact }
        for (key, tier) in table {
            if lower.hasPrefix(key) { return tier }
        }
        for (key, tier) in fallback {
            if lower.hasPrefix(key) { return tier }
        }
        return PriceTier(input: 0, output: 0, cache_write: 0, cache_read: 0)
    }

    static func cost(_ usage: ModelUsage, tier: PriceTier) -> Double {
        let m = 1_000_000.0
        return Double(usage.input_tokens)                   * tier.input       / m
             + Double(usage.output_tokens)                  * tier.output      / m
             + Double(usage.cache_creation_input_tokens)    * tier.cache_write / m
             + Double(usage.cache_read_input_tokens)        * tier.cache_read  / m
    }

    static func loadConfig(from dataDir: URL) -> AppConfig {
        let url = dataDir.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: url),
           let cfg = try? JSONDecoder().decode(AppConfig.self, from: data) {
            return cfg
        }
        return AppConfig(version: 1, store_project_paths: true, pricing: fallback)
    }
}
