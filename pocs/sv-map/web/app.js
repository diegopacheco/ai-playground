import { CATEGORIES, searchCompanies } from "./search.js";
import { clusterPoints } from "./cluster.js";
import { escapeHtml, logoHtml } from "./html.js";
import { createPalette } from "./palette.js";
import { createHelp } from "./help.js";

const VALLEY = [[37.23, -122.52], [37.81, -121.78]];
const TABS = [
  { id: "all", label: "All" },
  ...Object.entries(CATEGORIES).map(([id, c]) => ({ id, label: c.label, color: c.color })),
  { id: "directory", label: "Directory" }
];
const CLUSTER_RADIUS = 30;
const CLUSTER_UNTIL_ZOOM = 17;

const $ = id => document.getElementById(id);
const state = { companies: [], tab: "all", query: "", selected: null, rebuilding: false };

const map = L.map("map", { zoomControl: true, minZoom: 9, maxZoom: 19 }).fitBounds(VALLEY);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);
const markerLayer = L.layerGroup().addTo(map);

function categoryFilter() {
  return state.tab === "directory" ? "all" : state.tab;
}

function visibleCompanies() {
  return searchCompanies(state.companies, state.query, categoryFilter());
}

function osmLink(c) {
  return `https://www.openstreetmap.org/?mlat=${c.lat}&mlon=${c.lon}#map=18/${c.lat}/${c.lon}`;
}

function popupHtml(c) {
  const cat = CATEGORIES[c.category];
  return `<div class="popup">${logoHtml(c)}<div>
    <h3>${escapeHtml(c.name)}</h3>
    <span class="chip" style="color:${cat.color};border-color:${cat.color}">${cat.label}</span>
    <div class="address">${escapeHtml(c.address)}<br>${escapeHtml(c.city)}, CA</div>
    <div class="links"><a href="https://${escapeHtml(c.domain)}" target="_blank" rel="noopener">${escapeHtml(c.domain)}</a><a href="${osmLink(c)}" target="_blank" rel="noopener">Open in OpenStreetMap</a></div>
  </div></div>`;
}

function pinIcon(c) {
  const size = map.getZoom() >= 15 ? 42 : 32;
  return L.divIcon({
    className: `pin${state.selected === c.id ? " selected" : ""}`,
    html: logoHtml(c),
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -size / 2]
  });
}

function dominantColor(members) {
  const counts = {};
  for (const m of members) counts[m.company.category] = (counts[m.company.category] || 0) + 1;
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  return CATEGORIES[top].color;
}

function clusterIcon(members) {
  const size = Math.min(30 + Math.sqrt(members.length) * 4, 46);
  return L.divIcon({
    className: "pin",
    html: `<div class="cluster" style="background:${dominantColor(members)}">${members.length}</div>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2]
  });
}

function addPin(c) {
  const marker = L.marker([c.lat, c.lon], { icon: pinIcon(c), title: c.name, riseOnHover: true })
    .bindPopup(popupHtml(c), { maxWidth: 360 })
    .on("click", () => { state.selected = c.id; renderList(); })
    .addTo(markerLayer);
  if (state.selected === c.id) marker.openPopup();
}

function renderMarkers() {
  state.rebuilding = true;
  markerLayer.clearLayers();
  state.rebuilding = false;
  const points = visibleCompanies().map(company => ({ company, ...map.latLngToLayerPoint([company.lat, company.lon]) }));
  const radius = map.getZoom() >= CLUSTER_UNTIL_ZOOM ? 1 : CLUSTER_RADIUS;
  for (const group of clusterPoints(points, radius)) {
    const selectedInside = group.members.some(m => m.company.id === state.selected);
    if (group.members.length === 1 || selectedInside) {
      group.members.forEach(m => addPin(m.company));
      continue;
    }
    L.marker(map.layerPointToLatLng([group.x, group.y]), { icon: clusterIcon(group.members) })
      .on("click", () => map.fitBounds(L.latLngBounds(group.members.map(m => [m.company.lat, m.company.lon])), { padding: [60, 60], maxZoom: 18 }))
      .addTo(markerLayer);
  }
}

function renderList() {
  const hits = visibleCompanies();
  $("count").textContent = `${hits.length} of ${state.companies.length} companies`;
  $("list").innerHTML = hits.map(c =>
    `<li data-id="${c.id}" class="${state.selected === c.id ? "selected" : ""}">${logoHtml(c)}<div><div class="name">${escapeHtml(c.name)}</div><div class="city">${escapeHtml(c.address)} · ${escapeHtml(c.city)}</div></div></li>`).join("");
  $("list").querySelector(".selected")?.scrollIntoView({ block: "nearest" });
}

function renderDirectory() {
  const hits = visibleCompanies();
  $("directory").innerHTML = Object.entries(CATEGORIES).map(([id, cat]) => {
    const items = hits.filter(c => c.category === id);
    if (!items.length) return "";
    return `<h2><span class="chip" style="color:${cat.color};border-color:${cat.color}">${items.length}</span>${cat.label}</h2>
      <div class="grid">${items.map(c => `<div class="item" data-id="${c.id}">${logoHtml(c)}<div><div class="name">${escapeHtml(c.name)}</div><div class="addr">${escapeHtml(c.address)}, ${escapeHtml(c.city)}</div></div></div>`).join("")}</div>`;
  }).join("") || '<p class="count">No company matches.</p>';
}

function renderTabs() {
  $("tabs").innerHTML = TABS.map((t, i) => {
    const count = t.id === "all" || t.id === "directory" ? state.companies.length : state.companies.filter(c => c.category === t.id).length;
    const dot = t.color ? `<span class="dot" style="background:${t.color}"></span>` : "";
    return `<button class="tab${state.tab === t.id ? " active" : ""}" data-tab="${t.id}" title="⌘${i + 1}" type="button">${dot}${t.label}<small>${count}</small></button>`;
  }).join("");
}

function renderLegend() {
  $("legend").innerHTML = Object.values(CATEGORIES)
    .map(c => `<span class="chip" style="color:${c.color};border-color:${c.color}"><span class="dot" style="background:${c.color}"></span>${c.label}</span>`).join("");
}

function render() {
  const directory = state.tab === "directory";
  $("directory").hidden = !directory;
  $("map").style.visibility = directory ? "hidden" : "visible";
  renderTabs();
  renderList();
  if (directory) renderDirectory();
  else renderMarkers();
}

function setTab(id) {
  state.tab = id;
  const company = state.companies.find(c => c.id === state.selected);
  if (company && categoryFilter() !== "all" && company.category !== categoryFilter()) {
    state.selected = null;
    map.closePopup();
  }
  render();
  if (id !== "directory") map.invalidateSize();
}

function select(company) {
  state.selected = company.id;
  if (state.tab === "directory" || (categoryFilter() !== "all" && categoryFilter() !== company.category)) state.tab = "all";
  if (!searchCompanies(state.companies, state.query, categoryFilter()).includes(company)) {
    state.query = "";
    $("search").value = "";
  }
  render();
  map.invalidateSize();
  map.flyTo([company.lat, company.lon], Math.max(map.getZoom(), 17), { duration: 0.8 });
  map.once("moveend", renderMarkers);
}

function resetView() {
  state.selected = null;
  map.closePopup();
  map.flyToBounds(VALLEY, { duration: 0.6 });
  render();
}

function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 2600);
}

function wireEvents(palette, help) {
  map.on("zoomend", renderMarkers);
  map.on("popupclose", () => {
    if (!state.selected || state.rebuilding) return;
    state.selected = null;
    renderList();
  });
  $("search").addEventListener("input", event => { state.query = event.target.value; render(); });
  $("search").addEventListener("keydown", event => {
    if (event.key === "Escape" && event.target.value) { event.target.value = ""; state.query = ""; render(); event.stopPropagation(); }
    if (event.key === "Enter") { const first = visibleCompanies()[0]; if (first) select(first); }
  });
  $("tabs").addEventListener("click", event => { const tab = event.target.closest("[data-tab]"); if (tab) setTab(tab.dataset.tab); });
  const pickById = event => {
    const item = event.target.closest("[data-id]");
    if (item) select(state.companies.find(c => c.id === item.dataset.id));
  };
  $("list").addEventListener("click", pickById);
  $("directory").addEventListener("click", pickById);
  $("open-palette").addEventListener("click", () => palette.open());
  $("open-help").addEventListener("click", () => help.open());
  $("titlebar").addEventListener("dblclick", event => {
    if (event.target.closest("button")) return;
    window.svmap?.toggleMaximize();
  });

  document.addEventListener("keydown", event => {
    const cmd = event.metaKey || event.ctrlKey;
    if (cmd && event.key.toLowerCase() === "k") { event.preventDefault(); help.close(); palette.toggle(); return; }
    if (cmd && event.key === "/") { event.preventDefault(); palette.close(); help.toggle(); return; }
    if (cmd && event.key.toLowerCase() === "f") { event.preventDefault(); $("search").focus(); $("search").select(); return; }
    if (cmd && event.key.toLowerCase() === "r") { event.preventDefault(); resetView(); return; }
    if (cmd && /^[1-9]$/.test(event.key) && TABS[Number(event.key) - 1]) { event.preventDefault(); setTab(TABS[Number(event.key) - 1].id); return; }
    if (event.key === "Escape") { palette.close(); help.close(); }
  });

  window.svmap?.onToast(toast);
}

async function main() {
  state.companies = await (await fetch("/api/companies")).json();
  const palette = createPalette({
    overlay: $("palette"), input: $("palette-input"), list: $("palette-list"),
    companies: state.companies, logoHtml, onPick: company => select(company)
  });
  const help = createHelp({ overlay: $("help"), input: $("help-input"), grid: $("help-grid"), count: $("help-count") });
  renderLegend();
  wireEvents(palette, help);
  render();
}

main();
