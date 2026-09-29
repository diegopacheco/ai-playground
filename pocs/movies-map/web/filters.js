const EMPTY_FILTERS = { type: "", genre: "", decade: "" }

function decadeOf(year) {
  return year ? String(Math.floor(year / 10) * 10) : ""
}

function matchesFilters(movie, filters, skip) {
  if (skip !== "type" && filters.type && movie.type !== filters.type) return false
  if (skip !== "genre" && filters.genre && !(movie.categories || []).includes(filters.genre)) return false
  if (skip !== "decade" && filters.decade && decadeOf(movie.year) !== filters.decade) return false
  return true
}

function facetCounts(movies, filters, key) {
  const counts = new Map()
  for (const movie of movies) {
    if (!matchesFilters(movie, filters, key)) continue
    const values = key === "genre" ? movie.categories || [] : key === "decade" ? [decadeOf(movie.year)] : [movie.type]
    for (const value of values) if (value) counts.set(value, (counts.get(value) || 0) + 1)
  }
  return counts
}

if (typeof module !== "undefined") module.exports = { EMPTY_FILTERS, decadeOf, matchesFilters, facetCounts }
