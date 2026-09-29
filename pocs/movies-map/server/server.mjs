import http from "node:http"
import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { SF_BOX, CA_BOX, nearby, searchMovies, geocodeUrl, toPlaces, stats } from "./lib.mjs"

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")
const PORT = Number(process.env.API_PORT || 4545)
const DATA = path.join(ROOT, "data", "movies.json")
const STATIC = [
  ["/vendor/leaflet/", path.join(ROOT, "node_modules", "leaflet", "dist")],
  ["/", path.join(ROOT, "web")]
]
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml", ".json": "application/json" }
const HEADERS = { "User-Agent": "MoviesMap/1.0 (open-source SF film locations map)" }

const { movies } = JSON.parse(fs.readFileSync(DATA, "utf8"))
const byId = new Map(movies.map(m => [m.id, m]))
const geocodeCache = new Map()

function send(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store" })
  res.end(JSON.stringify(body))
}

async function geocode(text) {
  const key = text.trim().toLowerCase()
  if (geocodeCache.has(key)) return geocodeCache.get(key)
  let places = []
  for (const box of [SF_BOX, CA_BOX]) {
    const res = await fetch(geocodeUrl(text, box), { headers: HEADERS })
    if (!res.ok) throw new Error(`geocoder answered ${res.status}`)
    places = toPlaces(await res.json())
    if (places.length) break
  }
  geocodeCache.set(key, places)
  return places
}

function serveStatic(pathname, res) {
  for (const [prefix, dir] of STATIC) {
    if (!pathname.startsWith(prefix)) continue
    const rel = pathname.slice(prefix.length) || "index.html"
    const file = path.join(dir, rel)
    if (!file.startsWith(dir) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) continue
    res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" })
    fs.createReadStream(file).pipe(res)
    return true
  }
  return false
}

async function route(req, res) {
  const url = new URL(req.url, `http://localhost:${PORT}`)
  const p = url.pathname
  if (p === "/api/health") return send(res, 200, { status: "ok", ...stats(movies) })
  if (p === "/api/movies") return send(res, 200, movies)
  if (p.startsWith("/api/movies/")) {
    const movie = byId.get(decodeURIComponent(p.slice("/api/movies/".length)))
    return movie ? send(res, 200, movie) : send(res, 404, { error: "movie not found" })
  }
  if (p === "/api/search") return send(res, 200, searchMovies(movies, url.searchParams.get("q")))
  if (p === "/api/nearby") {
    const lat = Number(url.searchParams.get("lat"))
    const lng = Number(url.searchParams.get("lng"))
    const km = Number(url.searchParams.get("km") || 1)
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return send(res, 400, { error: "lat and lng are required" })
    return send(res, 200, nearby(movies, { lat, lng }, km))
  }
  if (p === "/api/geocode") {
    const q = (url.searchParams.get("q") || "").trim()
    if (!q) return send(res, 400, { error: "q is required" })
    return send(res, 200, await geocode(q))
  }
  if (req.method === "GET" && serveStatic(p, res)) return
  send(res, 404, { error: "not found" })
}

http.createServer((req, res) => {
  route(req, res).catch(err => send(res, 502, { error: err.message }))
}).listen(PORT, "127.0.0.1", () => console.log(`movies-map api on http://localhost:${PORT}`))
