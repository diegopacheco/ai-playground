const SHORTCUT_GROUPS = [
  {
    title: "Navigation",
    color: "#2563eb",
    icon: '<path d="M3 12h18M12 3l9 9-9 9"/>',
    items: [["⌘ 1", "Go to Map tab"], ["⌘ 2", "Go to Movies tab"], ["Esc", "Close details or modal"]]
  },
  {
    title: "Search",
    color: "#ea580c",
    icon: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>',
    items: [["⌘ K", "Search movies, actors, genres, places"], ["↑ ↓", "Move through results"], ["Enter", "Go to selected result"], ["⌘ /", "Show all shortcuts"]]
  },
  {
    title: "View",
    color: "#16a34a",
    icon: '<rect x="3" y="5" width="18" height="14" rx="3"/><path d="M8 10h8M12 6v8"/>',
    items: [["⌘ +", "Zoom in"], ["⌘ -", "Zoom out"], ["⌘ 0", "Reset zoom"], ["⌘ ⇧ Enter", "Toggle full screen"], ["Double click title bar", "Maximize or restore window"]]
  },
  {
    title: "Edit",
    color: "#9333ea",
    icon: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
    items: [["⌘ C", "Copy"], ["⌘ V", "Paste"], ["⌘ X", "Cut"], ["⌘ A", "Select all"]]
  },
  {
    title: "Capture",
    color: "#db2777",
    icon: '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="4"/>',
    items: [["⌘ P", "Save a screenshot of the app to Desktop"]]
  }
]

function filterShortcutGroups(groups, text) {
  const q = text.trim().toLowerCase()
  if (!q) return groups
  const out = []
  for (const group of groups) {
    if (group.title.toLowerCase().includes(q)) {
      out.push(group)
      continue
    }
    const items = group.items.filter(([keys, label]) => keys.toLowerCase().includes(q) || label.toLowerCase().includes(q))
    if (items.length) out.push({ ...group, items })
  }
  return out
}

if (typeof module !== "undefined") module.exports = { SHORTCUT_GROUPS, filterShortcutGroups }
