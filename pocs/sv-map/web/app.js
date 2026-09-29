import { CATEGORIES, searchCompanies } from "./search.js";
import { spreadPoints } from "./spread.js";
import { escapeHtml, logoHtml } from "./html.js";
import { createPalette } from "./palette.js";
import { createHelp } from "./help.js";

const VALLEY = [[37.23, -122.52], [37.81, -121.78]];
const TABS = [
  { id: "all", label: "All" },
  ...Object.entries(CATEGORIES).map(([id, c]) => ({ id, label: c.label, color: c.color })),
  { id: "directory", label: "Directory" }
];
const PIN_GAP = 2;

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

function pinSize() {
  const zoom = map.getZoom();
  if (zoom >= 15) return 42;
  if (zoom >= 13) return 34;
  return 28;
}

function pinIcon(c, size) {
  return L.divIcon({
    className: `pin${state.selected === c.id ? " selected" : ""}`,
    html: logoHtml(c),
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -size / 2]
  });
}

function addPin(c, latLng, size) {
  const marker = L.marker(latLng, { icon: pinIcon(c, size), title: c.name, riseOnHover: true })
    .bindPopup(popupHtml(c), { maxWidth: 360 })
    .on("click", () => { state.selected = c.id; renderList(); })
    .addTo(markerLayer);
  if (state.selected === c.id) marker.openPopup();
}

function addLeader(c, anchor, placed) {
  const color = CATEGORIES[c.category].color;
  L.polyline([anchor, placed], { color, weight: 1.5, opacity: 0.8, interactive: false }).addTo(markerLayer);
  L.circleMarker(anchor, { radius: 3, color: "#fff", weight: 1, fillColor: color, fillOpacity: 1, interactive: false }).addTo(markerLayer);
}

function renderMarkers() {
  state.rebuilding = true;
  markerLayer.clearLayers();
  state.rebuilding = false;
  const size = pinSize();
  const points = visibleCompanies().map(company => ({ company, ...map.latLngToLayerPoint([company.lat, company.lon]) }));
  for (const p of spreadPoints(points, size + PIN_GAP)) {
    const anchor = L.latLng(p.company.lat, p.company.lon);
    const placed = map.layerPointToLatLng([p.x, p.y]);
    if (Math.hypot(p.x - p.anchorX, p.y - p.anchorY) > 4) addLeader(p.company, anchor, placed);
    addPin(p.company, placed, size);
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
