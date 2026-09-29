const ICONS = {
  search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>',
  tabs: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 9h18M9 5v4"/>',
  view: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
  map: '<path d="M9 4L3 6v14l6-2 6 2 6-2V4l-6 2z"/><path d="M9 4v14M15 6v14"/>',
  edit: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
  capture: '<path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="3"/>'
};

export const SHORTCUT_GROUPS = [
  { title: "Search", icon: "search", color: "#7c3aed", rows: [
    [["⌘", "K"], "Search companies and go to one"],
    [["⌘", "F"], "Focus the sidebar search"],
    [["↑", "↓", "↵"], "Move and open in search results"],
    [["Esc"], "Clear search or close modal"]
  ] },
  { title: "Tabs", icon: "tabs", color: "#2563eb", rows: [
    [["⌘", "1"], "All companies"],
    [["⌘", "2"], "Big Tech"],
    [["⌘", "3"], "Tech"],
    [["⌘", "4"], "AI Labs"],
    [["⌘", "5"], "AI Startups"],
    [["⌘", "6"], "Directory"]
  ] },
  { title: "View", icon: "view", color: "#059669", rows: [
    [["⌘", "+"], "Zoom the app in"],
    [["⌘", "-"], "Zoom the app out"],
    [["⌘", "0"], "Reset app zoom"],
    [["⌘", "⇧", "↵"], "Toggle full screen"],
    [["Double click title"], "Maximize or restore the window"],
    [["⌘", "/"], "Show this shortcuts panel"]
  ] },
  { title: "Map", icon: "map", color: "#ea580c", rows: [
    [["Click logo"], "Show the company address"],
    [["Click bubble"], "Zoom into a group of companies"],
    [["Scroll"], "Zoom the map"],
    [["Drag"], "Pan the map"],
    [["⇧", "Drag"], "Zoom into a box"],
    [["⌘", "R"], "Reset the map to the whole valley"]
  ] },
  { title: "Edit", icon: "edit", color: "#db2777", rows: [
    [["⌘", "C"], "Copy"],
    [["⌘", "X"], "Cut"],
    [["⌘", "V"], "Paste"],
    [["⌘", "A"], "Select all"]
  ] },
  { title: "Capture", icon: "capture", color: "#0891b2", rows: [
    [["⌘", "P"], "Save a screenshot of the window to the Desktop"]
  ] }
];

export function filterGroups(groups, text) {
  const query = String(text || "").trim().toLowerCase();
  if (!query) return groups;
  return groups
    .map(group => group.title.toLowerCase().includes(query)
      ? group
      : { ...group, rows: group.rows.filter(([keys, desc]) => `${keys.join(" ")} ${desc}`.toLowerCase().includes(query)) })
    .filter(group => group.rows.length > 0);
}

function renderGroup(group) {
  const rows = group.rows.map(([keys, desc]) =>
    `<div class="help-row"><span>${desc}</span><span>${keys.map(k => `<kbd>${k}</kbd>`).join(" ")}</span></div>`).join("");
  const icon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[group.icon]}</svg>`;
  return `<section class="help-group" style="--group:${group.color}"><h4>${icon}${group.title}</h4>${rows}</section>`;
}

export function createHelp({ overlay, input, grid, count }) {
  const render = () => {
    const groups = filterGroups(SHORTCUT_GROUPS, input.value);
    const total = groups.reduce((sum, g) => sum + g.rows.length, 0);
    count.textContent = `${total} shortcut${total === 1 ? "" : "s"}`;
    grid.innerHTML = total ? groups.map(renderGroup).join("") : '<div class="help-empty">No shortcut matches that search.</div>';
  };
  const close = () => { overlay.hidden = true; };
  const open = () => { overlay.hidden = false; input.value = ""; render(); input.focus(); };
  input.addEventListener("input", render);
  input.addEventListener("keydown", event => {
    if (event.key !== "Escape") return;
    event.preventDefault();
    event.stopPropagation();
    if (input.value) { input.value = ""; render(); } else close();
  });
  overlay.addEventListener("mousedown", event => { if (event.target === overlay) close(); });
  return { open, close, toggle: () => (overlay.hidden ? open() : close()), isOpen: () => !overlay.hidden };
}
