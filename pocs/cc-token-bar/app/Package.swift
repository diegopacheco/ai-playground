// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "cc-token-bar",
    platforms: [.macOS(.v13)],
    targets: [
        .target(
            name: "CCMetrics",
            path: "Sources/CCMetrics"
        ),
        .executableTarget(
            name: "cc-token-bar",
            dependencies: ["CCMetrics"],
            path: "Sources/cc-token-bar"
        ),
        .executableTarget(
            name: "cc-metrics-test",
            dependencies: ["CCMetrics"],
            path: "Sources/cc-metrics-test"
        )
    ]
)
