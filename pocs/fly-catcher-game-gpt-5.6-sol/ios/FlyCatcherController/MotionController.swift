import CoreMotion
import Foundation
import Network

final class MotionController: ObservableObject {
    @Published var host = UserDefaults.standard.string(forKey: "gameHost") ?? ""
    @Published var port = UserDefaults.standard.string(forKey: "gamePort") ?? "5005"
    @Published private(set) var active = false
    @Published private(set) var status = "Enter the Mac address"
    @Published private(set) var acceleration = CMAcceleration(x: 0, y: 0, z: 0)

    private let motion = CMMotionManager()
    private let queue = DispatchQueue(label: "fly-catcher.udp")
    private var connection: NWConnection?
    private var lastSnap = Date.distantPast

    func connect(from url: URL) {
        guard url.scheme?.lowercased() == "flycatcher",
              url.host?.lowercased() == "connect",
              let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
              let scannedHost = components.queryItems?.first(where: { $0.name == "host" })?.value,
              let scannedPort = components.queryItems?.first(where: { $0.name == "port" })?.value else {
            status = "Pairing code is invalid"
            return
        }
        host = scannedHost
        port = scannedPort
        start()
    }

    func start() {
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedHost.isEmpty else {
            status = "Mac address is required"
            return
        }
        guard let number = UInt16(port), let networkPort = NWEndpoint.Port(rawValue: number) else {
            status = "Port must be 1 through 65535"
            return
        }
        stop()
        UserDefaults.standard.set(trimmedHost, forKey: "gameHost")
        UserDefaults.standard.set(port, forKey: "gamePort")
        let link = NWConnection(host: NWEndpoint.Host(trimmedHost), port: networkPort, using: .udp)
        connection = link
        link.stateUpdateHandler = { [weak self] state in
            DispatchQueue.main.async {
                guard let self else { return }
                guard self.connection === link else { return }
                switch state {
                case .ready:
                    self.active = true
                    self.status = "Linked to the kitchen"
                    self.startMotion()
                case .waiting(let error):
                    self.active = false
                    self.status = error.localizedDescription
                case .failed(let error):
                    self.active = false
                    self.status = error.localizedDescription
                    self.stopMotion()
                case .cancelled:
                    self.active = false
                    self.status = "Controller stopped"
                    self.stopMotion()
                default:
                    self.status = "Opening local UDP link"
                }
            }
        }
        link.start(queue: queue)
    }

    func stop() {
        stopMotion()
        connection?.cancel()
        connection = nil
        active = false
    }

    func sendSnap() {
        lastSnap = Date()
        send(["type": "snap"])
    }

    private func startMotion() {
        guard motion.isAccelerometerAvailable else {
            status = "Accelerometer unavailable"
            return
        }
        motion.accelerometerUpdateInterval = 1.0 / 30.0
        motion.startAccelerometerUpdates(to: .main) { [weak self] reading, _ in
            guard let self, let reading else { return }
            let value = reading.acceleration
            self.acceleration = value
            self.send([
                "type": "motion",
                "ax": value.x,
                "ay": value.y,
                "az": value.z
            ])
            let force = sqrt(value.x * value.x + value.y * value.y + value.z * value.z)
            if force > 1.72 && Date().timeIntervalSince(self.lastSnap) > 0.55 {
                self.sendSnap()
            }
        }
    }

    private func stopMotion() {
        motion.stopAccelerometerUpdates()
    }

    private func send(_ object: [String: Any]) {
        guard let connection, let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        connection.send(content: data, completion: .contentProcessed { _ in })
    }
}
