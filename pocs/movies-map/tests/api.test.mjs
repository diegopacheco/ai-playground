import test from "node:test"
import assert from "node:assert/strict"
import { spawn } from "node:child_process"
import fs from "node:fs"

const PORT = 4599
const base = `http://localhost:${PORT}`
const { movies } = JSON.parse(fs.readFileSync(new URL("../data/movies.json", import.meta.url), "utf8"))
let server

test.before(async () => {
  server = spawn("node", ["server/server.mjs"], { cwd: new URL("..", import.meta.url), env: { ...process.env, API_PORT: String(PORT) } })
  for (let i = 0; i < 50; i++) {
    if (await fetch(`${base}/api/health`).then(r => r.ok).catch(() => false)) return
    await new Promise(r => setTimeout(r, 100))
  }
  throw new Error("server did not start")
})

test.after(() => server.kill())

const get = path => fetch(`${base}${path}`).then(async r => ({ status: r.status, body: await r.json() }))

test("the built dataset covers San Francisco with posters for most titles", () => {
  assert.ok(movies.length > 250)
  const inSf = movies.flatMap(m => m.locations).filter(l => l.lat > 37.6 && l.lat < 37.9 && l.lng > -122.6 && l.lng < -122.3)
  assert.ok(inSf.length / movies.flatMap(m => m.locations).length > 0.95)
  assert.ok(movies.filter(m => m.poster).length / movies.length > 0.7)
})

test("health reports the loaded data so the boot screen can show it", async () => {
  const { status, body } = await get("/api/health")
  assert.equal(status, 200)
  assert.equal(body.movies, movies.length)
})

test("a movie detail has the fields the poster click shows", async () => {
  const { body } = await get(`/api/movies/${movies[0].id}`)
  for (const key of ["title", "year", "actors", "genres", "description", "locations"]) assert.ok(key in body, key)
})

test("the Golden Gate Bridge has movies shot next to it", async () => {
  const { body } = await get("/api/nearby?lat=37.8199&lng=-122.4783&km=1")
  assert.ok(body.length > 5)
  assert.ok(body.every((h, i) => i === 0 || body[i - 1].distanceKm <= h.distanceKm))
})

test("bad requests are rejected with clear errors", async () => {
  assert.equal((await get("/api/nearby?lat=abc")).status, 400)
  assert.equal((await get("/api/geocode")).status, 400)
  assert.equal((await get("/api/movies/nope")).status, 404)
})

test("the ui and the map library are served", async () => {
  const html = await fetch(`${base}/`).then(r => r.text())
  assert.match(html, /Movies Map/)
  assert.equal((await fetch(`${base}/vendor/leaflet/leaflet.js`)).status, 200)
})
