import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private var statusItem: NSStatusItem!
    private var panelWindow: NSPanel!
    private var hostingController: NSHostingController<PanelView>!
    private var visualEffect: NSVisualEffectView!
    private let store = DataStore()
    private var eventMonitor: Any?
    private let panelWidth: CGFloat = 360
    private let preferredPanelHeight: CGFloat = 720
    private let menuBarGap: CGFloat = 8
    private let screenBottomMargin: CGFloat = 12

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

        let initial = NSRect(x: 0, y: 0, width: panelWidth, height: 480)
        panelWindow = NSPanel(
            contentRect: initial,
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

        visualEffect = NSVisualEffectView(frame: initial)
        visualEffect.material = .popover
        visualEffect.blendingMode = .behindWindow
        visualEffect.state = .active
        visualEffect.wantsLayer = true
        visualEffect.layer?.cornerRadius = 12
        visualEffect.layer?.masksToBounds = true
        visualEffect.autoresizingMask = [.width, .height]

        hostingController = NSHostingController(rootView: PanelView(store: store))
        hostingController.view.frame = visualEffect.bounds
        hostingController.view.autoresizingMask = [.width, .height]
        visualEffect.addSubview(hostingController.view)

        panelWindow.contentView = visualEffect
    }

    private func fittedPanelSize(maxHeight: CGFloat) -> NSSize {
        let height = min(preferredPanelHeight, maxHeight)
        return NSSize(width: panelWidth, height: max(240, height))
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
        let screen = buttonWindow.screen ?? NSScreen.main
        let visible = screen?.visibleFrame ?? .zero
        let topLimit = min(buttonFrameOnScreen.minY - menuBarGap, visible.maxY - menuBarGap)
        let maxHeight = max(240, topLimit - visible.minY - screenBottomMargin)
        let size = fittedPanelSize(maxHeight: maxHeight)
        var originX = buttonFrameOnScreen.midX - size.width / 2
        originX = min(max(originX, visible.minX + 8), visible.maxX - size.width - 8)
        let originY = topLimit - size.height
        let frame = NSRect(x: originX, y: originY, width: size.width, height: size.height)
        panelWindow.setFrame(frame, display: true)
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
