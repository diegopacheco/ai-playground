// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "cc-token-bar",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "cc-token-bar",
            path: "Sources/cc-token-bar"
        )
    ]
)
