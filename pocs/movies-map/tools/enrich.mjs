export const SCREEN_WORDS = /\b(film|movie|series|television|documentary|miniseries|sitcom|drama)\b/i

export function toNumber(value) {
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

export function movieKey(title, year) {
  return `${title.trim().toLowerCase()}|${year || ""}`
}

export function slug(title, year) {
  return `${title}-${year || ""}`.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")
}

export function groupRows(rows) {
  const movies = new Map()
  for (const row of rows) {
    if (!row.title) continue
    const lat = toNumber(row.latitude)
    const lng = toNumber(row.longitude)
    if (lat === null || lng === null) continue
    const year = toNumber(row.release_year)
    const key = movieKey(row.title, year)
    if (!movies.has(key)) {
      movies.set(key, {
        id: slug(row.title.trim(), year),
        title: row.title.trim(),
        year,
        director: row.director || "",
        writer: row.writer || "",
        productionCompany: row.production_company || "",
        distributor: row.distributor || "",
        actors: [],
        locations: []
      })
    }
    const movie = movies.get(key)
    for (const actor of [row.actor_1, row.actor_2, row.actor_3]) {
      if (actor && !movie.actors.includes(actor.trim())) movie.actors.push(actor.trim())
    }
    const name = (row.locations || "Unnamed location").trim()
    if (!movie.locations.some(l => l.name === name && l.lat === lat && l.lng === lng)) {
      movie.locations.push({ name, lat, lng, neighborhood: row.analysis_neighborhood || "", funFact: row.fun_facts || "" })
    }
  }
  return [...movies.values()].sort((a, b) => a.title.localeCompare(b.title))
}

export function claimIds(entity, property) {
  return (entity?.claims?.[property] || [])
    .map(c => c.mainsnak?.datavalue?.value?.id)
    .filter(Boolean)
}

export function claimYear(entity) {
  for (const property of ["P577", "P580"]) {
    for (const c of entity?.claims?.[property] || []) {
      const time = c.mainsnak?.datavalue?.value?.time
      const match = time && /^[+-](\d{4})/.exec(time)
      if (match) return Number(match[1])
    }
  }
  return null
}

export function isScreenWork(entity, year) {
  const description = entity?.descriptions?.en?.value || ""
  if (!SCREEN_WORDS.test(description)) return false
  const released = claimYear(entity)
  if (year === null || released === null) return true
  return Math.abs(released - year) <= 2
}

export function mergeActors(fromDataset, fromWikidata, limit = 10) {
  const all = [...fromDataset]
  for (const name of fromWikidata) {
    if (!all.some(a => a.toLowerCase() === name.toLowerCase())) all.push(name)
  }
  return all.slice(0, limit)
}

export function searchTitle(title) {
  return title.replace(/\s*[-–:,]?\s*\b(season|ep|episode|pilot|part)\b.*$/i, "").replace(/["“”]/g, "").trim() || title
}

const TV_TITLE = /\b(season|ep|episode|pilot)\b/i
const TV_TEXT = /\b(television|tv|web|streaming) (series|show|program|programme|sitcom|miniseries|drama|anthology)|\bsitcom\b|\bminiseries\b|\bdocuseries\b|\bseason of\b|\btelevision network\b/i

export function titleType({ title, genres = [], description = "" }) {
  if (TV_TITLE.test(title)) return "tv"
  if (genres.some(g => /television/i.test(g))) return "tv"
  const firstSentence = description.split(/(?<=\.)\s/)[0] || ""
  const tvAt = firstSentence.search(TV_TEXT)
  const filmAt = firstSentence.search(/\b(film|movie)\b/i)
  return tvAt >= 0 && (filmAt < 0 || tvAt < filmAt) ? "tv" : "movie"
}

const GENRE_RULES = [
  ["Drama", /drama|melodrama|trial film/i],
  ["Comedy", /comedy|parody|sitcom/i],
  ["Action", /action|martial arts|heist|superhero|buddy cop|disaster|kaiju|spy/i],
  ["Thriller", /thriller|noir/i],
  ["Crime", /crime|heist|police procedural|detective|vigilante|noir|prison/i],
  ["Mystery", /mystery|detective/i],
  ["Romance", /romance|romantic|remarriage/i],
  ["Science Fiction", /science fiction|cyberpunk|time-travel|dystopian|apocalyptic|alien|speculative|kaiju/i],
  ["Fantasy", /fantasy|ghost|vampire/i],
  ["Horror", /horror|zombie|monster|vampire|ghost/i],
  ["Adventure", /adventure|treasure hunt|road movie|sword-and-sandal/i],
  ["Biography & History", /biographical|historical|docudrama/i],
  ["Documentary", /documentary|docudrama|docuseries/i],
  ["Musical", /musical/i],
  ["Family", /children|family|teen|coming-of-age/i],
  ["War", /\bwar film/i],
  ["LGBTQ+", /lgbt/i]
]

export function broadGenres(genres = []) {
  return GENRE_RULES.filter(([, rule]) => genres.some(g => rule.test(g))).map(([name]) => name)
}
