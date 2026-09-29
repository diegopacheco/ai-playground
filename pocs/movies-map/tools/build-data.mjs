import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { groupRows, claimIds, isScreenWork, mergeActors, searchTitle } from "./enrich.mjs"

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")
const OUT = path.join(ROOT, "data", "movies.json")
const CACHE = path.join(ROOT, "data", "wiki-cache.json")
const SOURCE = "https://data.sfgov.org/resource/yitu-d5am.json?$limit=50000"
const WIKI = "https://en.wikipedia.org/w/api.php"
const WIKIDATA = "https://www.wikidata.org/w/api.php"
const HEADERS = { "User-Agent": "MoviesMap/1.0 (open-source SF film locations map; https://github.com/diegopacheco/ai-playground)" }

const sleep = ms => new Promise(r => setTimeout(r, ms))

async function getJson(url, tries = 6) {
  for (let i = 1; ; i++) {
    const res = await fetch(url, { headers: HEADERS })
    if (res.ok) return res.json()
    if (i >= tries) throw new Error(`${res.status} ${url}`)
    const wait = Number(res.headers.get("retry-after")) || 2 * i
    await sleep(wait * 1000)
  }
}

function query(base, params) {
  return `${base}?${new URLSearchParams({ format: "json", formatversion: "2", ...params })}`
}

async function searchCandidates(movie) {
  const name = searchTitle(movie.title)
  const terms = [`${name} ${movie.year || ""} film`, `${name} television series`]
  const titles = []
  for (const term of terms) {
    const data = await getJson(query(WIKI, { action: "query", list: "search", srsearch: term, srlimit: "5" }))
    for (const hit of data.query?.search || []) if (!titles.includes(hit.title)) titles.push(hit.title)
  }
  if (!titles.length) return []
  const pages = await getJson(query(WIKI, { action: "query", prop: "pageprops", ppprop: "wikibase_item", titles: titles.join("|") }))
  const byTitle = new Map((pages.query?.pages || []).map(p => [p.title, p.pageprops?.wikibase_item]))
  return titles.map(t => ({ title: t, qid: byTitle.get(t) })).filter(c => c.qid)
}

async function entities(ids, props = "claims|descriptions|labels") {
  const out = {}
  for (let i = 0; i < ids.length; i += 50) {
    const data = await getJson(query(WIKIDATA, { action: "wbgetentities", ids: ids.slice(i, i + 50).join("|"), props, languages: "en" }))
    Object.assign(out, data.entities || {})
  }
  return out
}

function normalized(text) {
  return text.toLowerCase().replace(/\(.*?\)/g, "").replace(/[^a-z0-9]+/g, " ").trim()
}

async function matchMovie(movie) {
  const candidates = await searchCandidates(movie)
  if (!candidates.length) return null
  const found = await entities(candidates.map(c => c.qid))
  const wanted = normalized(searchTitle(movie.title))
  const year = searchTitle(movie.title) === movie.title ? movie.year : null
  const valid = candidates.filter(c => isScreenWork(found[c.qid], year))
  const pick = valid.find(c => normalized(c.title) === wanted) || valid.find(c => normalized(c.title).includes(wanted)) || null
  return pick ? { page: pick.title, entity: found[pick.qid] } : null
}

async function summary(page) {
  const data = await getJson(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(page.replace(/ /g, "_"))}`)
  return {
    description: data.extract || "",
    poster: data.thumbnail?.source || "",
    posterLarge: data.originalimage?.source || "",
    wikipedia: data.content_urls?.desktop?.page || ""
  }
}

async function pool(items, size, fn) {
  let next = 0
  let done = 0
  const worker = async () => {
    while (next < items.length) {
      const index = next++
      await fn(items[index])
      done++
      if (done % 25 === 0) console.log(`enriched ${done}/${items.length}`)
    }
  }
  await Promise.all(Array.from({ length: size }, worker))
}

async function main() {
  console.log("fetching DataSF film locations")
  const rows = await getJson(SOURCE)
  const movies = groupRows(rows)
  console.log(`grouped ${rows.length} rows into ${movies.length} titles`)
  const cache = fs.existsSync(CACHE) ? JSON.parse(fs.readFileSync(CACHE, "utf8")) : {}
  const labelIds = new Set()
  const matched = new Map()
  await pool(movies, 2, async movie => {
    try {
      if (!(movie.id in cache)) {
        const match = await matchMovie(movie)
        cache[movie.id] = match ? { page: match.page, genres: claimIds(match.entity, "P136"), cast: claimIds(match.entity, "P161").slice(0, 10), ...(await summary(match.page)) } : null
        fs.writeFileSync(CACHE, JSON.stringify(cache))
        await sleep(300)
      }
      const hit = cache[movie.id]
      if (!hit) return
      matched.set(movie.id, hit)
      for (const id of [...hit.genres, ...hit.cast]) labelIds.add(id)
      Object.assign(movie, { description: hit.description, poster: hit.poster, posterLarge: hit.posterLarge, wikipedia: hit.wikipedia })
    } catch (err) {
      console.error(`skip ${movie.title}: ${err.message}`)
    }
  })
  const labels = await entities([...labelIds], "labels")
  const label = id => labels[id]?.labels?.en?.value
  for (const movie of movies) {
    const match = matched.get(movie.id)
    movie.genres = match ? match.genres.map(label).filter(Boolean).slice(0, 5) : []
    movie.actors = mergeActors(movie.actors, match ? match.cast.map(label).filter(Boolean) : [])
    movie.description = movie.description || ""
    movie.poster = movie.poster || ""
    movie.posterLarge = movie.posterLarge || ""
    movie.wikipedia = movie.wikipedia || ""
  }
  fs.mkdirSync(path.dirname(OUT), { recursive: true })
  fs.writeFileSync(OUT, JSON.stringify({ source: "DataSF Film Locations in San Francisco", builtAt: new Date().toISOString(), movies }, null, 1))
  const withPoster = movies.filter(m => m.poster).length
  console.log(`wrote ${movies.length} titles, ${withPoster} with posters, to data/movies.json`)
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
