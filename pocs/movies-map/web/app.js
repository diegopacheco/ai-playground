const SF_CENTER = [37.7749, -122.4294]
const CELL = 110
const SF_BOUNDS = L.latLngBounds([37.70, -122.53], [37.84, -122.35])
const $ = id => document.getElementById(id)

const state = { movies: [], byId: new Map(), points: [], address: null, activeTab: "map", focusId: null, filters: { ...EMPTY_FILTERS } }

const visible = movie => matchesFilters(movie, state.filters)
const escapeHtml = text => String(text ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c])

const spots = n => `${n} location${n === 1 ? "" : "s"}`

function hue(text) {
  let h = 0
  for (const ch of text) h = (h * 31 + ch.charCodeAt(0)) % 360
  return h
}

function posterHtml(movie, cls, lazy = false) {
  const fallback = `<div class="${cls} fallback" style="background:linear-gradient(160deg,hsl(${hue(movie.title)},70%,62%),hsl(${(hue(movie.title) + 40) % 360},65%,45%))">${escapeHtml(movie.title)}</div>`
  if (!movie.poster) return fallback
  return `<img class="${cls}" src="${escapeHtml(movie.poster)}" alt="${escapeHtml(movie.title)} poster"${lazy ? ' loading="lazy"' : ""} data-fallback="${escapeHtml(fallback)}">`
}

document.addEventListener("error", e => {
  const img = e.target
  if (img.tagName === "IMG" && img.dataset.fallback) img.outerHTML = img.dataset.fallback
}, true)

async function api(path) {
  const res = await fetch(path)
  const body = await res.json()
  if (!res.ok) throw new Error(body.error || res.statusText)
  return body
}

function toast(text) {
  const el = $("toast")
  el.textContent = text
  el.hidden = false
  clearTimeout(toast.timer)
  toast.timer = setTimeout(() => (el.hidden = true), 2600)
}

const map = L.map("map", { zoomControl: true, minZoom: 5, maxZoom: 19 }).setView(SF_CENTER, 13)
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  maxZoom: 19
}).addTo(map)
const posterLayer = L.layerGroup().addTo(map)
const addressLayer = L.layerGroup().addTo(map)

function clusterPoints() {
  const zoom = map.getZoom()
  const bounds = map.getBounds().pad(0.3)
  const cells = new Map()
  for (const point of state.points) {
    if (state.focusId ? point.movie.id !== state.focusId : !visible(point.movie)) continue
    if (!bounds.contains(point.latlng)) continue
    const px = map.project(point.latlng, zoom)
    const key = `${Math.floor(px.x / CELL)}:${Math.floor(px.y / CELL)}`
    if (!cells.has(key)) cells.set(key, [])
    cells.get(key).push(point)
  }
  return [...cells.values()]
}

function renderPosters() {
  posterLayer.clearLayers()
  for (const group of clusterPoints()) {
    const movieIds = [...new Set(group.map(p => p.movie.id))]
    const lead = group.find(p => p.movie.poster) || group[0]
    const lat = group.reduce((s, p) => s + p.latlng.lat, 0) / group.length
    const lng = group.reduce((s, p) => s + p.latlng.lng, 0) / group.length
    const count = movieIds.length > 1 ? `<span class="count">${movieIds.length}</span>` : ""
    const icon = L.divIcon({
      className: "poster-pin",
      html: `<div class="frame">${posterHtml(lead.movie, "")}${count}</div>`,
      iconSize: [44, 64],
      iconAnchor: [22, 75]
    })
    const marker = L.marker([lat, lng], { icon, title: movieIds.length > 1 ? `${movieIds.length} movies` : lead.movie.title })
    marker.on("click", () => clickGroup(group, movieIds, marker))
    posterLayer.addLayer(marker)
  }
}

function clickGroup(group, movieIds, marker) {
  if (movieIds.length === 1) return openDetail(movieIds[0], group[0].location)
  if (map.getZoom() < 17) {
    const bounds = L.latLngBounds(group.map(p => p.latlng))
    const target = Math.min(map.getBoundsZoom(bounds, false, [80, 80]), 18)
    return target > map.getZoom() ? map.flyToBounds(bounds, { padding: [80, 80], maxZoom: 18 }) : map.flyTo(bounds.getCenter(), map.getZoom() + 2)
  }
  const list = movieIds.map(id => {
    const movie = state.byId.get(id)
    return `<li data-id="${escapeHtml(id)}">${posterHtml(movie, "thumb")}<div><strong>${escapeHtml(movie.title)}</strong><br><small>${movie.year || ""}</small></div></li>`
  }).join("")
  marker.bindPopup(`<ul class="pick">${list}</ul>`, { offset: [0, -70] }).openPopup()
  marker.getPopup().getElement().querySelectorAll("li").forEach(li => li.addEventListener("click", () => {
    map.closePopup()
    openDetail(li.dataset.id, group.find(p => p.movie.id === li.dataset.id).location)
  }))
}

map.on("moveend zoomend", renderPosters)
map.on("click", () => {
  if (state.focusId) closeDetail()
})

function focusMovie(movie) {
  state.focusId = movie.id
  showTab("map")
  $("focus-title").textContent = `${movie.title}${movie.year ? ` (${movie.year})` : ""}`
  $("focus-banner").hidden = false
  map.closePopup()
  renderPosters()
  const inSf = movie.locations.filter(l => SF_BOUNDS.contains([l.lat, l.lng]))
  const bounds = L.latLngBounds((inSf.length ? inSf : movie.locations).map(l => [l.lat, l.lng]))
  map.flyToBounds(bounds, { paddingTopLeft: [60, 80], paddingBottomRight: [460, 60], maxZoom: 16 })
}

function clearFocus() {
  if (!state.focusId) return
  state.focusId = null
  $("focus-banner").hidden = true
  renderPosters()
}

function resultItem(movie, subtitle) {
  return `<li data-id="${escapeHtml(movie.id)}">${posterHtml(movie, "thumb")}<div class="meta"><strong>${escapeHtml(movie.title)}</strong><span>${escapeHtml(subtitle)}</span></div></li>`
}

function renderList(title, items) {
  $("list-title").textContent = title
  $("results").innerHTML = items.length ? items.join("") : '<li class="empty">No titles match here. Try a larger radius or fewer filters.</li>'
}

function renderDefaultList() {
  const top = state.movies.filter(visible).sort((a, b) => b.locations.length - a.locations.length).slice(0, 30)
  renderList(state.movies.every(visible) ? "Most filmed in San Francisco" : "Most filmed matching your filters", top.map(m => resultItem(m, `${m.year || ""} · ${spots(m.locations.length)}`)))
}

$("results").addEventListener("click", e => {
  const li = e.target.closest("li[data-id]")
  if (!li) return
  const movie = state.byId.get(li.dataset.id)
  const spot = li.dataset.spot ? movie.locations[Number(li.dataset.spot)] : movie.locations[0]
  openDetail(movie.id, spot)
  map.flyTo([spot.lat, spot.lng], 16)
})

async function findAddress(text) {
  const places = $("places")
  places.innerHTML = '<span class="error" style="color:var(--muted)">Searching…</span>'
  try {
    const found = await api(`/api/geocode?q=${encodeURIComponent(text)}`)
    if (!found.length) {
      places.innerHTML = '<span class="error">No address found in California for that search.</span>'
      return
    }
    places.innerHTML = found.map((p, i) => `<button data-i="${i}">${escapeHtml(p.label)}</button>`).join("")
    places.querySelectorAll("button").forEach(b => b.addEventListener("click", () => setAddress(found[Number(b.dataset.i)], b)))
    setAddress(found[0], places.querySelector("button"))
  } catch (err) {
    places.innerHTML = `<span class="error">Address search failed: ${escapeHtml(err.message)}</span>`
  }
}

async function setAddress(place, button) {
  state.address = place
  $("places").querySelectorAll("button").forEach(b => b.classList.toggle("active", b === button))
  await refreshNearby(true)
}

async function refreshNearby(fly) {
  const place = state.address
  if (!place) return
  const km = Number($("radius").value)
  addressLayer.clearLayers()
  L.circle([place.lat, place.lng], { radius: km * 1000, color: "#3b82f6", weight: 1, fillOpacity: 0.06 }).addTo(addressLayer)
  L.marker([place.lat, place.lng], { icon: L.divIcon({ className: "", html: '<div class="address-pin"></div>', iconSize: [18, 18], iconAnchor: [9, 9] }), zIndexOffset: -1000 }).addTo(addressLayer)
  if (fly) map.flyToBounds(L.latLng(place.lat, place.lng).toBounds(km * 2000), { maxZoom: 17 })
  const hits = (await api(`/api/nearby?lat=${place.lat}&lng=${place.lng}&km=${km}`)).filter(h => visible(state.byId.get(h.id)))
  const short = place.label.split(",").slice(0, 2).join(",")
  renderList(`${hits.length} movies near ${short}`, hits.map(h => {
    const movie = state.byId.get(h.id)
    const index = movie.locations.indexOf(movie.locations.find(l => l.lat === h.location.lat && l.lng === h.location.lng && l.name === h.location.name))
    const meters = h.distanceKm < 1 ? `${Math.round(h.distanceKm * 1000)} m` : `${h.distanceKm.toFixed(1)} km`
    return resultItem(movie, `${meters} · ${h.location.name}`).replace("<li ", `<li data-spot="${index}" `)
  }))
}

$("address-form").addEventListener("submit", e => {
  e.preventDefault()
  const text = $("address").value.trim()
  if (text) findAddress(text)
})
$("radius").addEventListener("change", () => refreshNearby(true))
$("clear-address").addEventListener("click", () => {
  state.address = null
  $("address").value = ""
  $("places").innerHTML = ""
  addressLayer.clearLayers()
  renderDefaultList()
})

function openDetail(id, focus) {
  const movie = state.byId.get(id)
  if (!movie) return
  const facts = [["Director", movie.director], ["Writer", movie.writer], ["Studio", movie.productionCompany], ["Distributor", movie.distributor]]
    .filter(([, v]) => v).map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(v)}</dd>`).join("")
  const spotItems = movie.locations.map((l, i) => `<li data-spot="${i}"${l === focus ? ' style="border-color:var(--accent)"' : ""}>${escapeHtml(l.name)}${l.neighborhood ? `<small>${escapeHtml(l.neighborhood)}</small>` : ""}${l.funFact ? `<em>${escapeHtml(l.funFact)}</em>` : ""}</li>`).join("")
  $("detail").innerHTML = `
    <button class="close" title="Close (Esc)">&times;</button>
    <div class="hero">
      ${movie.posterLarge ? posterHtml({ ...movie, poster: movie.posterLarge }, "poster") : posterHtml(movie, "poster")}
      <div>
        <h2>${escapeHtml(movie.title)}</h2>
        <div class="year">${movie.year || "Year unknown"}</div>
        <div class="chips"><span class="chip type">${movie.type === "tv" ? "TV show" : "Movie"}</span></div>
        <div class="chips">${(movie.categories || []).map(g => `<span class="chip genre">${escapeHtml(g)}</span>`).join("") || '<span class="chip">Genre unknown</span>'}</div>
      </div>
    </div>
    <div class="body">
      <h4>Description</h4>
      <p>${escapeHtml(movie.description || "No description available.")}</p>
      <h4>Actors</h4>
      <div class="chips">${movie.actors.map(a => `<span class="chip">${escapeHtml(a)}</span>`).join("") || "<p>Unknown</p>"}</div>
      <h4>Credits</h4>
      <dl class="facts">${facts}</dl>
      <h4>Filmed at ${spots(movie.locations.length)} in San Francisco</h4>
      <ul class="spots">${spotItems}</ul>
      ${movie.wikipedia ? `<h4>More</h4><p><a href="${escapeHtml(movie.wikipedia)}" target="_blank" rel="noopener">Read on Wikipedia</a></p>` : ""}
    </div>`
  focusMovie(movie)
  $("detail").classList.add("open")
  $("detail").setAttribute("aria-hidden", "false")
  $("detail").scrollTop = 0
  $("detail").querySelector(".close").addEventListener("click", closeDetail)
  $("detail").querySelectorAll(".spots li").forEach(li => li.addEventListener("click", () => {
    const spot = movie.locations[Number(li.dataset.spot)]
    showTab("map")
    map.flyTo([spot.lat, spot.lng], 17)
  }))
}

function closeDetail() {
  $("detail").classList.remove("open")
  $("detail").setAttribute("aria-hidden", "true")
  clearFocus()
}

function showTab(name) {
  state.activeTab = name
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name))
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.id === `tab-${name}`))
  if (name === "map") setTimeout(() => map.invalidateSize(), 0)
}

$("focus-clear").addEventListener("click", closeDetail)

$("tabs").addEventListener("click", e => {
  const b = e.target.closest("button[data-tab]")
  if (b) showTab(b.dataset.tab)
})

function renderGrid() {
  const q = $("grid-filter").value.trim().toLowerCase()
  const shown = state.movies.filter(m => {
    if (!visible(m)) return false
    if (!q) return true
    return [m.title, m.director, ...(m.actors || []), ...(m.genres || [])].some(v => (v || "").toLowerCase().includes(q))
  })
  $("grid-count").textContent = `${shown.length} shown`
  $("grid").innerHTML = shown.map(m => `<div class="tile" data-id="${escapeHtml(m.id)}">${posterHtml(m, "poster", true)}<strong>${escapeHtml(m.title)}</strong><span>${m.year || ""} · ${spots(m.locations.length)}</span></div>`).join("")
}

$("grid-filter").addEventListener("input", renderGrid)
$("grid").addEventListener("click", e => {
  const tile = e.target.closest(".tile")
  if (tile) openDetail(tile.dataset.id)
})

const search = { items: [], index: 0, timer: null }

function openSearch() {
  closeHelp()
  $("search-modal").hidden = false
  $("search-input").value = ""
  $("search-input").focus()
  runSearch()
}

function closeSearch() {
  $("search-modal").hidden = true
}

async function runSearch() {
  const q = $("search-input").value.trim()
  const items = []
  if (q) {
    const movies = await api(`/api/search?q=${encodeURIComponent(q)}`)
    for (const m of movies.slice(0, 12)) items.push({ kind: "Movie", html: `${posterHtml(m, "thumb")}<div><strong>${escapeHtml(m.title)}</strong><br><small>${m.year || ""} · ${escapeHtml((m.genres || []).slice(0, 2).join(", "))}</small></div>`, go: () => openDetail(m.id) })
    const lower = q.toLowerCase()
    const seen = new Set()
    for (const p of state.points) {
      if (items.length > 20) break
      if (!p.location.name.toLowerCase().includes(lower) || seen.has(p.location.name)) continue
      seen.add(p.location.name)
      items.push({ kind: "Place", html: `<span class="icon">&#9679;</span><div><strong>${escapeHtml(p.location.name)}</strong><br><small>${escapeHtml(p.movie.title)}</small></div>`, go: () => { openDetail(p.movie.id, p.location); map.flyTo(p.latlng, 17) } })
    }
    items.push({ kind: "Address", html: `<span class="icon">&#8982;</span><div><strong>Find movies near “${escapeHtml(q)}”</strong><br><small>Look up this address in San Francisco or California</small></div>`, go: () => { showTab("map"); $("address").value = q; findAddress(q) } })
  } else {
    items.push({ kind: "Tab", html: '<span class="icon">1</span><div><strong>Map</strong></div>', go: () => showTab("map") })
    items.push({ kind: "Tab", html: '<span class="icon">2</span><div><strong>Movies</strong></div>', go: () => showTab("movies") })
  }
  if ($("search-input").value.trim() !== q) return
  search.items = items
  search.index = 0
  paintSearch()
}

function paintSearch() {
  $("search-results").innerHTML = search.items.map((it, i) => `<li data-i="${i}" class="${i === search.index ? "active" : ""}">${it.html}<span class="kind">${it.kind}</span></li>`).join("")
  $("search-results").querySelector("li.active")?.scrollIntoView({ block: "nearest" })
}

function pickSearch(i) {
  const item = search.items[i]
  if (!item) return
  closeSearch()
  item.go()
}

$("search-input").addEventListener("input", () => {
  clearTimeout(search.timer)
  search.timer = setTimeout(runSearch, 120)
})
$("search-input").addEventListener("keydown", e => {
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault()
    search.index = (search.index + (e.key === "ArrowDown" ? 1 : -1) + search.items.length) % search.items.length
    paintSearch()
  } else if (e.key === "Enter") {
    e.preventDefault()
    pickSearch(search.index)
  }
})
$("search-results").addEventListener("click", e => {
  const li = e.target.closest("li[data-i]")
  if (li) pickSearch(Number(li.dataset.i))
})

function openHelp() {
  closeSearch()
  $("help-modal").hidden = false
  $("help-input").value = ""
  paintHelp()
  $("help-input").focus()
}

function closeHelp() {
  $("help-modal").hidden = true
}

function paintHelp() {
  const groups = filterShortcutGroups(SHORTCUT_GROUPS, $("help-input").value)
  const total = groups.reduce((n, g) => n + g.items.length, 0)
  $("help-count").textContent = `${total} shortcut${total === 1 ? "" : "s"}`
  $("help-groups").innerHTML = groups.length ? groups.map(g => `
    <section class="group" style="--c:${g.color}">
      <h5><svg viewBox="0 0 24 24">${g.icon}</svg>${escapeHtml(g.title)}</h5>
      ${g.items.map(([keys, label]) => `<div class="item"><span>${escapeHtml(label)}</span><kbd>${escapeHtml(keys)}</kbd></div>`).join("")}
    </section>`).join("") : '<p class="help-empty">No shortcut matches your search.</p>'
}

$("help-input").addEventListener("input", paintHelp)
$("open-search").addEventListener("click", openSearch)
$("open-help").addEventListener("click", openHelp)
document.querySelectorAll(".modal").forEach(m => m.addEventListener("mousedown", e => {
  if (e.target === m) m.hidden = true
}))

document.addEventListener("keydown", e => {
  const mod = e.metaKey || e.ctrlKey
  if (mod && e.key.toLowerCase() === "k") {
    e.preventDefault()
    return $("search-modal").hidden ? openSearch() : closeSearch()
  }
  if (mod && e.key === "/") {
    e.preventDefault()
    return $("help-modal").hidden ? openHelp() : closeHelp()
  }
  if (mod && (e.key === "1" || e.key === "2")) {
    e.preventDefault()
    return showTab(e.key === "1" ? "map" : "movies")
  }
  if (e.key === "Escape") {
    if (!$("help-modal").hidden) {
      if ($("help-input").value) {
        $("help-input").value = ""
        return paintHelp()
      }
      return closeHelp()
    }
    if (!$("search-modal").hidden) return closeSearch()
    closeDetail()
  }
})

$("titlebar").addEventListener("dblclick", e => {
  if (e.target.closest("button, input")) return
  window.moviesMap?.toggleMaximize?.()
})

window.moviesMap?.onToast?.(toast)

async function loadMovies() {
  for (let attempt = 1; ; attempt++) {
    try {
      return await api("/api/movies")
    } catch (err) {
      if (attempt >= 5) throw err
      await new Promise(r => setTimeout(r, 1000))
    }
  }
}

function renderFilterControls() {
  const f = state.filters
  const types = facetCounts(state.movies, f, "type")
  const all = state.movies.filter(m => matchesFilters(m, f, "type")).length
  $("filter-type").querySelectorAll("button").forEach(b => {
    const label = { "": "All", movie: "Movies", tv: "TV shows" }[b.dataset.type]
    b.innerHTML = `${label}<small>${b.dataset.type ? types.get(b.dataset.type) || 0 : all}</small>`
    b.classList.toggle("active", b.dataset.type === f.type)
  })
  const genres = facetCounts(state.movies, f, "genre")
  const genreNames = [...new Set([...genres.keys(), f.genre].filter(Boolean))].sort((a, b) => (genres.get(b) || 0) - (genres.get(a) || 0) || a.localeCompare(b))
  $("filter-genre").innerHTML = '<option value="">All genres</option>' + genreNames.map(g => `<option value="${escapeHtml(g)}">${escapeHtml(g)} (${genres.get(g) || 0})</option>`).join("")
  $("filter-genre").value = f.genre
  const decades = facetCounts(state.movies, f, "decade")
  const decadeNames = [...new Set([...decades.keys(), f.decade].filter(Boolean))].sort()
  $("filter-decade").innerHTML = '<option value="">All decades</option>' + decadeNames.map(d => `<option value="${d}">${d}s (${decades.get(d) || 0})</option>`).join("")
  $("filter-decade").value = f.decade
  const shown = state.movies.filter(visible).length
  $("filter-count").textContent = `${shown} of ${state.movies.length} titles`
  $("filter-reset").hidden = !f.type && !f.genre && !f.decade
}

function applyFilters() {
  renderFilterControls()
  renderPosters()
  if (state.address) refreshNearby(false)
  else renderDefaultList()
  renderGrid()
}

function setFilter(key, value) {
  state.filters = { ...state.filters, [key]: value }
  applyFilters()
}

$("filter-type").addEventListener("click", e => {
  const b = e.target.closest("button[data-type]")
  if (b) setFilter("type", b.dataset.type)
})
$("filter-genre").addEventListener("change", e => setFilter("genre", e.target.value))
$("filter-decade").addEventListener("change", e => setFilter("decade", e.target.value))
$("filter-reset").addEventListener("click", () => {
  state.filters = { ...EMPTY_FILTERS }
  applyFilters()
})

function showLoadError(err) {
  $("list-title").textContent = "Movies could not be loaded"
  $("results").innerHTML = `<li class="empty">${escapeHtml(err.message)}. Is the API running? <button class="link" id="retry-load">Retry</button></li>`
  $("retry-load").addEventListener("click", () => location.reload())
}

async function boot() {
  state.movies = await loadMovies()
  state.byId = new Map(state.movies.map(m => [m.id, m]))
  state.points = state.movies.flatMap(movie => movie.locations.map(location => ({ movie, location, latlng: L.latLng(location.lat, location.lng) })))
  applyFilters()
}

boot().catch(showLoadError)
