// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "genie-speech-cli",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "genie-speech-cli",
            path: "Sources"
        )
    ]
)
