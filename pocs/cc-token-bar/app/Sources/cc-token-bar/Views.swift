import SwiftUI
import Charts
import AppKit

enum PanelTab: Hashable {
    case cost
    case latency
}

struct PanelView: View {
    @ObservedObject var store: DataStore
    @State private var tab: PanelTab = .cost

    var body: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(alignment: .leading, spacing: 0) {
                header
                tabBar
                if tab == .cost {
                    kpiSection
                    divider
                    cacheSection
                    divider
                    chartSection
                    divider
                    toolsSection
                    divider
                    modelsSection
                } else {
                    latencySection
                }
                footer
            }
            .padding(.top, 12)
            .padding(.bottom, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(width: 360)
    }

    private var tabBar: some View {
        Picker("", selection: $tab) {
            Text("Cost").tag(PanelTab.cost)
            Text("Latency").tag(PanelTab.latency)
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .padding(.horizontal, 14).padding(.vertical, 10)
    }

    private var header: some View {
        HStack {
            Text("cc-token-bar").font(.system(size: 13, weight: .semibold))
            Spacer()
            Button(action: { NSApp.terminate(nil) }) {
                Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
            }.buttonStyle(.plain)
        }
        .padding(.horizontal, 14).padding(.bottom, 10)
        .overlay(Divider(), alignment: .bottom)
    }

    private var divider: some View {
        Divider().opacity(0.6)
    }

    private var kpiSection: some View {
        HStack(spacing: 10) {
            kpi(title: "Today",
                value: DataStore.formatTokens(store.agg.today.total),
                sub: "\(DataStore.formatUSD(store.agg.today.costUSD))  ·  \(DataStore.formatCount(store.agg.sessionsToday)) sessions")
            kpi(title: "All time",
                value: DataStore.formatTokens(store.agg.lifetime.total),
                sub: "\(DataStore.formatUSD(store.agg.lifetime.costUSD))  ·  \(DataStore.formatCount(store.agg.sessionsLifetime)) sessions")
        }
        .padding(14)
    }

    private func kpi(title: String, value: String, sub: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.system(size: 11)).foregroundStyle(.secondary)
            Text(value).font(.system(size: 17, weight: .semibold)).monospacedDigit()
            Text(sub).font(.system(size: 11)).foregroundStyle(.secondary).monospacedDigit()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.quaternary.opacity(0.4))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var cacheSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionTitle("Cache hit ratio")
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(String(format: "%.1f%%", store.agg.cacheHitRatio * 100))
                    .font(.system(size: 22, weight: .bold))
                    .foregroundStyle(store.agg.cacheHitRatio >= 0.6
                        ? Color(red: 0.18, green: 0.62, blue: 0.45)
                        : Color(red: 0.92, green: 0.55, blue: 0.20))
                    .monospacedDigit()
                Text("cache reads vs total input").font(.system(size: 11)).foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
    }

    private var chartSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionTitle("Last 7 days — input / output")
            Chart {
                ForEach(store.agg.byDay) { day in
                    BarMark(
                        x: .value("Day", day.label),
                        y: .value("Input", day.input)
                    ).foregroundStyle(Color.accentColor).cornerRadius(2)
                    BarMark(
                        x: .value("Day", day.label),
                        y: .value("Output", day.output)
                    ).foregroundStyle(Color.orange).cornerRadius(2)
                }
            }
            .chartYAxis(.hidden)
            .chartXAxis {
                AxisMarks(values: .automatic) {
                    AxisValueLabel().font(.system(size: 10)).foregroundStyle(.secondary)
                }
            }
            .frame(height: 110)
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
    }

    private var toolsSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle("Tools (by cost)")
            if store.agg.tools.isEmpty {
                Text("No tool data yet").font(.system(size: 11)).foregroundStyle(.secondary)
            } else {
                let maxCost = store.agg.tools.first?.costUSD ?? 1
                ForEach(store.agg.tools) { t in
                    HStack(spacing: 10) {
                        Text(t.name).font(.system(size: 12, weight: .medium))
                            .lineLimit(1).truncationMode(.tail)
                            .frame(width: 96, alignment: .leading)
                        bar(fraction: maxCost > 0 ? t.costUSD / maxCost : 0)
                        Text(DataStore.formatUSD(t.costUSD))
                            .font(.system(size: 11)).monospacedDigit()
                            .frame(width: 60, alignment: .trailing)
                        Text("\(DataStore.formatCount(t.count))×")
                            .font(.system(size: 11)).foregroundStyle(.secondary).monospacedDigit()
                            .frame(width: 48, alignment: .trailing)
                    }
                    .padding(.vertical, 3)
                }
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
    }

    private var latencySection: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle("Tool latency (avg per call)")
            if store.agg.toolLatencies.isEmpty {
                Text("No latency data yet").font(.system(size: 11)).foregroundStyle(.secondary)
            } else {
                let maxMs = store.agg.toolLatencies.first?.avgMs ?? 1
                ForEach(store.agg.toolLatencies) { t in
                    HStack(spacing: 10) {
                        Text(t.name).font(.system(size: 12, weight: .medium))
                            .lineLimit(1).truncationMode(.tail)
                            .frame(width: 96, alignment: .leading)
                        bar(fraction: maxMs > 0 ? t.avgMs / maxMs : 0)
                        Text(DataStore.formatMs(t.avgMs))
                            .font(.system(size: 11)).monospacedDigit()
                            .frame(width: 60, alignment: .trailing)
                        Text("\(DataStore.formatCount(t.count))×")
                            .font(.system(size: 11)).foregroundStyle(.secondary).monospacedDigit()
                            .frame(width: 48, alignment: .trailing)
                    }
                    .padding(.vertical, 3)
                }
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
    }

    private func bar(fraction: Double) -> some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 3).fill(.quaternary)
                RoundedRectangle(cornerRadius: 3)
                    .fill(LinearGradient(colors: [Color.accentColor, .orange],
                                         startPoint: .leading, endPoint: .trailing))
                    .frame(width: max(2, geo.size.width * fraction))
            }
        }
        .frame(height: 6)
    }

    private var modelsSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle("Cost by model")
            if store.agg.byModel.isEmpty {
                Text("No model data yet").font(.system(size: 11)).foregroundStyle(.secondary)
            } else {
                let total = store.agg.byModel.reduce(0.0) { $0 + $1.1.costUSD }
                ForEach(store.agg.byModel, id: \.0) { (name, t) in
                    HStack {
                        Text(name).font(.system(size: 12)).foregroundStyle(.secondary)
                        Spacer()
                        let pct = total > 0 ? t.costUSD / total * 100 : 0
                        Text("\(DataStore.formatUSD(t.costUSD))  ·  \(String(format: "%.0f%%", pct))")
                            .font(.system(size: 12)).monospacedDigit()
                    }
                    .padding(.vertical, 2)
                }
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
    }

    private var footer: some View {
        HStack {
            Button("Open data folder") {
                let url = FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent(".cc-token-bar")
                NSWorkspace.shared.open(url)
            }.buttonStyle(.plain).foregroundStyle(.secondary).font(.system(size: 11))
            Spacer()
            Button("Quit") { NSApp.terminate(nil) }
                .buttonStyle(.plain).foregroundStyle(.secondary).font(.system(size: 11))
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .overlay(Divider(), alignment: .top)
    }

    private func sectionTitle(_ s: String) -> some View {
        Text(s.uppercased())
            .font(.system(size: 10, weight: .semibold))
            .foregroundStyle(.secondary)
            .kerning(0.6)
    }
}
