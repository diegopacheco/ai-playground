import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate, NSPopoverDelegate {
    private var statusItem: NSStatusItem!
    private var popover: NSPopover!
    private let store = DataStore()

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
            button.action = #selector(togglePopover(_:))
            button.sendAction(on: [.leftMouseUp, .rightMouseUp])
        }

        popover = NSPopover()
        popover.behavior = .transient
        popover.delegate = self
        popover.contentSize = NSSize(width: 360, height: 640)
        popover.contentViewController = NSHostingController(
            rootView: PanelView(store: store)
        )
    }

    @objc func togglePopover(_ sender: Any?) {
        guard let button = statusItem.button else { return }
        if popover.isShown {
            popover.performClose(sender)
        } else {
            store.refreshNow()
            popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            popover.contentViewController?.view.window?.makeKey()
        }
    }

    func popoverDidShow(_ notification: Notification) {
        store.startVisibleRefresh()
    }

    func popoverDidClose(_ notification: Notification) {
        store.stopVisibleRefresh()
    }
}
