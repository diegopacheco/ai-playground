import SwiftUI

@main
struct FlyCatcherControllerApp: App {
    @StateObject private var controller = MotionController()

    var body: some Scene {
        WindowGroup {
            ControllerView()
                .environmentObject(controller)
                .onOpenURL { url in
                    controller.connect(from: url)
                }
        }
    }
}
