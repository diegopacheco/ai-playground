import test from "node:test"
import assert from "node:assert/strict"
import fs from "node:fs"
import vm from "node:vm"

const context = { module: { exports: {} } }
vm.runInNewContext(fs.readFileSync(new URL("../web/filters.js", import.meta.url), "utf8"), context)
const { EMPTY_FILTERS, matchesFilters, facetCounts, decadeOf } = context.module.exports

const movies = [
  { id: "vertigo", type: "movie", year: 1958, categories: ["Thriller", "Mystery"] },
  { id: "bullitt", type: "movie", year: 1968, categories: ["Action", "Thriller"] },
  { id: "looking", type: "tv", year: 2014, categories: ["Comedy", "Drama"] },
  { id: "devs", type: "tv", year: 2020, categories: ["Thriller", "Science Fiction"] }
]
const ids = f => movies.filter(m => matchesFilters(m, { ...EMPTY_FILTERS, ...f })).map(m => m.id)

test("no filters shows every title", () => {
  assert.deepEqual(ids({}), ["vertigo", "bullitt", "looking", "devs"])
})

test("type, genre and decade combine so only titles matching all of them stay on the map", () => {
  assert.deepEqual(ids({ type: "tv" }), ["looking", "devs"])
  assert.deepEqual(ids({ genre: "Thriller" }), ["vertigo", "bullitt", "devs"])
  assert.deepEqual(ids({ type: "movie", genre: "Thriller", decade: "1960" }), ["bullitt"])
  assert.deepEqual(ids({ type: "tv", genre: "Action" }), [])
})

test("genre counts follow the other filters so a choice never leads to an empty map", () => {
  const counts = facetCounts(movies, { ...EMPTY_FILTERS, type: "tv", genre: "Thriller" }, "genre")
  assert.equal(counts.get("Thriller"), 1)
  assert.equal(counts.get("Comedy"), 1)
  assert.equal(counts.has("Action"), false)
})

test("decades group years by ten", () => {
  assert.equal(decadeOf(1958), "1950")
  assert.equal(decadeOf(null), "")
})
