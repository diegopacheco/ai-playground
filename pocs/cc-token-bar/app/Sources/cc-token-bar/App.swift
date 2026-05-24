import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private var statusItem: NSStatusItem!
    private var panelWindow: NSPanel!
    private let store = DataStore()
    private var eventMonitor: Any?
    private let panelSize = NSSize(width: 360, height: 640)
    private let menuBarGap: CGFloat = 12

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        store.start()

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem.button {
            let config = NSImage.SymbolConfiguration(pointSize: 14, weight: .semibold)
            let icon = NSImage(systemSymbolName: "chart.bar.xaxis.ascending",
                               accessibilityDescription: "cc-token-bar")?
                .withSymbolConfiguration(config)
            icon?.isTemplate = true
            button.image = icon
            button.imagePosition = .imageLeft
            button.title = " cc"
            button.toolTip = "cc-token-bar — Claude Code usage"
            button.target = self
            button.action = #selector(togglePanel(_:))
            button.sendAction(on: [.leftMouseUp, .rightMouseUp])
        }

        panelWindow = NSPanel(
            contentRect: NSRect(origin: .zero, size: panelSize),
            styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        panelWindow.isFloatingPanel = true
        panelWindow.level = .statusBar
        panelWindow.hasShadow = true
        panelWindow.isOpaque = false
        panelWindow.backgroundColor = .clear
        panelWindow.hidesOnDeactivate = false
        panelWindow.delegate = self
        panelWindow.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]

        let container = NSVisualEffectView(frame: NSRect(origin: .zero, size: panelSize))
        container.material = .popover
        container.blendingMode = .behindWindow
        container.state = .active
        container.wantsLayer = true
        container.layer?.cornerRadius = 12
        container.layer?.masksToBounds = true
        container.autoresizingMask = [.width, .height]

        let host = NSHostingController(rootView: PanelView(store: store))
        host.view.frame = container.bounds
        host.view.autoresizingMask = [.width, .height]
        container.addSubview(host.view)

        panelWindow.contentView = container
    }

    @objc func togglePanel(_ sender: Any?) {
        if panelWindow.isVisible {
            hidePanel()
        } else {
            showPanel()
        }
    }

    private func showPanel() {
        guard let button = statusItem.button,
              let buttonWindow = button.window else { return }
        store.refreshNow()
        let buttonFrameOnScreen = buttonWindow.convertToScreen(button.convert(button.bounds, to: nil))
        let originX = buttonFrameOnScreen.midX - panelSize.width / 2
        let originY = buttonFrameOnScreen.minY - menuBarGap - panelSize.height
        panelWindow.setFrameOrigin(NSPoint(x: originX, y: originY))
        panelWindow.orderFrontRegardless()
        store.startVisibleRefresh()
        installDismissMonitor()
    }

    private func hidePanel() {
        panelWindow.orderOut(nil)
        store.stopVisibleRefresh()
        removeDismissMonitor()
    }

    private func installDismissMonitor() {
        removeDismissMonitor()
        eventMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] _ in
            self?.hidePanel()
        }
    }

    private func removeDismissMonitor() {
        if let m = eventMonitor {
            NSEvent.removeMonitor(m)
            eventMonitor = nil
        }
    }

    func windowDidResignKey(_ notification: Notification) {}
}
