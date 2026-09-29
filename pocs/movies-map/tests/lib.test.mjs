import test from "node:test"
import assert from "node:assert/strict"
import { distanceKm, nearby, searchMovies, geocodeUrl, toPlaces, SF_BOX } from "../server/lib.mjs"

const movies = [
  { id: "vertigo-1958", title: "Vertigo", year: 1958, director: "Alfred Hitchcock", genres: ["thriller film"], actors: ["James Stewart"], locations: [{ name: "Fort Point", lat: 37.8106, lng: -122.4771 }] },
  { id: "bullitt-1968", title: "Bullitt", year: 1968, director: "Peter Yates", genres: ["action film"], actors: ["Steve McQueen"], locations: [{ name: "Russian Hill", lat: 37.8014, lng: -122.4187 }, { name: "Fort Point parking", lat: 37.8100, lng: -122.4760 }] },
  { id: "milk-2008", title: "Milk", year: 2008, director: "Gus Van Sant", genres: ["biographical film"], actors: ["Sean Penn"], locations: [{ name: "Castro Street", lat: 37.7609, lng: -122.4350 }] }
]

test("distance between the Ferry Building and Coit Tower is about 1 km", () => {
  const d = distanceKm({ lat: 37.7955, lng: -122.3937 }, { lat: 37.8024, lng: -122.4058 })
  assert.ok(d > 1 && d < 1.5, `got ${d}`)
})

test("nearby lists only movies inside the radius, closest first, once per movie", () => {
  const hits = nearby(movies, { lat: 37.8106, lng: -122.4771 }, 1)
  assert.deepEqual(hits.map(h => h.id), ["vertigo-1958", "bullitt-1968"])
  assert.equal(hits[1].location.name, "Fort Point parking")
})

test("a larger radius around the Castro still excludes the Golden Gate shoots", () => {
  assert.deepEqual(nearby(movies, { lat: 37.7609, lng: -122.4350 }, 2).map(h => h.id), ["milk-2008"])
})

test("search ranks title matches above actor matches", () => {
  assert.equal(searchMovies(movies, "milk")[0].id, "milk-2008")
  assert.equal(searchMovies(movies, "mcqueen")[0].id, "bullitt-1968")
  assert.equal(searchMovies(movies, "thriller")[0].id, "vertigo-1958")
  assert.deepEqual(searchMovies(movies, "   "), [])
})

test("geocoding is bounded to the box so addresses resolve in San Francisco first", () => {
  const url = new URL(geocodeUrl("Lombard Street", SF_BOX))
  assert.equal(url.searchParams.get("bounded"), "1")
  assert.equal(url.searchParams.get("viewbox"), "-122.53,37.84,-122.35,37.7")
})

test("street segments with the same label show once so the user picks between real places", () => {
  const places = toPlaces([{ display_name: "Lombard Street, SF", lat: "37.80", lon: "-122.41" }, { display_name: "Lombard Street, SF", lat: "37.79", lon: "-122.43" }, { display_name: "Lombard, Oakland", lat: "37.8", lon: "-122.2" }])
  assert.deepEqual(places.map(p => p.label), ["Lombard Street, SF", "Lombard, Oakland"])
  assert.equal(places[0].lat, 37.8)
})
