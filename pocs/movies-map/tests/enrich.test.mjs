import test from "node:test"
import assert from "node:assert/strict"
import { groupRows, isScreenWork, mergeActors, claimYear, searchTitle, titleType, broadGenres } from "../tools/enrich.mjs"

const row = (over = {}) => ({ title: "Vertigo", release_year: "1958", locations: "Fort Point", latitude: "37.81", longitude: "-122.47", actor_1: "James Stewart", actor_2: "Kim Novak", ...over })

test("rows of one movie collapse into a single title so the map shows one poster per film", () => {
  const movies = groupRows([row(), row({ locations: "Mission Dolores", latitude: "37.76", longitude: "-122.43", actor_3: "Barbara Bel Geddes" })])
  assert.equal(movies.length, 1)
  assert.equal(movies[0].locations.length, 2)
  assert.deepEqual(movies[0].actors, ["James Stewart", "Kim Novak", "Barbara Bel Geddes"])
})

test("remakes with the same title stay separate movies because the year differs", () => {
  const movies = groupRows([row({ title: "The Parent Trap", release_year: "1961" }), row({ title: "The Parent Trap", release_year: "1998" })])
  assert.equal(movies.length, 2)
  assert.notEqual(movies[0].id, movies[1].id)
})

test("rows without coordinates are dropped since they cannot be plotted", () => {
  assert.equal(groupRows([row({ latitude: undefined })]).length, 0)
})

test("the same shoot location listed twice is kept once so pins do not stack", () => {
  assert.equal(groupRows([row(), row()])[0].locations.length, 1)
})

const entity = (description, year) => ({ descriptions: { en: { value: description } }, claims: { P577: [{ mainsnak: { datavalue: { value: { time: `+${year}-05-09T00:00:00Z` } } } }] } })

test("a wikidata match must be a screen work from about the same year to avoid wrong posters", () => {
  assert.equal(isScreenWork(entity("1958 film by Alfred Hitchcock", 1958), 1958), true)
  assert.equal(isScreenWork(entity("1954 novel by Boileau-Narcejac", 1954), 1958), false)
  assert.equal(isScreenWork(entity("2012 film", 2012), 1958), false)
  assert.equal(claimYear(entity("film", 1958)), 1958)
})

test("dataset actors come first and wikidata cast only fills the gaps", () => {
  assert.deepEqual(mergeActors(["Sean Penn"], ["sean penn", "Josh Brolin"]), ["Sean Penn", "Josh Brolin"])
})

test("tv episode rows search for the show so episodes still get the show poster", () => {
  assert.equal(searchTitle("Chance - Season 1 ep105"), "Chance")
  assert.equal(searchTitle("Looking Season 2 ep 202"), "Looking")
  assert.equal(searchTitle("Budding Prospects, Pilot"), "Budding Prospects")
  assert.equal(searchTitle("Vertigo"), "Vertigo")
})

test("episodes and series are tv shows so the type filter separates them from films", () => {
  assert.equal(titleType({ title: "Chance - Season 1 ep105" }), "tv")
  assert.equal(titleType({ title: "Looking", genres: ["LGBT-related television series"] }), "tv")
  assert.equal(titleType({ title: "Devs", description: "Devs is an American science fiction thriller television miniseries created by Alex Garland." }), "tv")
  assert.equal(titleType({ title: "Parks and Recreation", description: "The sixth season of Parks and Recreation originally aired on the NBC television network." }), "tv")
})

test("a film based on a tv series is still a movie", () => {
  assert.equal(titleType({ title: "Star Trek II", description: "Star Trek II: The Wrath of Khan is a 1982 American science fiction film based on the television series Star Trek." }), "movie")
  assert.equal(titleType({ title: "Dirty Harry", description: "Dirty Harry is a 1971 American action-thriller film, the first in the Dirty Harry series." }), "movie")
  assert.equal(titleType({ title: "Vertigo" }), "movie")
})

test("fine grained wikidata genres collapse into a short list people can pick from", () => {
  assert.deepEqual(broadGenres(["psychological thriller film", "romance film"]), ["Thriller", "Romance"])
  assert.deepEqual(broadGenres(["drama television series", "LGBT-related television series"]), ["Drama", "LGBTQ+"])
  assert.deepEqual(broadGenres(["buddy cop film"]), ["Action"])
  assert.deepEqual(broadGenres([]), [])
})
