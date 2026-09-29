export const SF_BOX = { west: -122.53, south: 37.70, east: -122.35, north: 37.84 }
export const CA_BOX = { west: -124.48, south: 32.53, east: -114.13, north: 42.01 }

export function distanceKm(a, b) {
  const rad = d => (d * Math.PI) / 180
  const dLat = rad(b.lat - a.lat)
  const dLng = rad(b.lng - a.lng)
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dLng / 2) ** 2
  return 6371 * 2 * Math.asin(Math.sqrt(h))
}

export function nearby(movies, point, km = 1, limit = 150) {
  const hits = []
  for (const movie of movies) {
    let best = null
    for (const location of movie.locations) {
      const d = distanceKm(point, location)
      if (d <= km && (!best || d < best.distanceKm)) best = { location, distanceKm: d }
    }
    if (best) hits.push({ id: movie.id, title: movie.title, year: movie.year, poster: movie.poster, ...best })
  }
  return hits.sort((a, b) => a.distanceKm - b.distanceKm).slice(0, limit)
}

export function searchMovies(movies, text, limit = 40) {
  const q = (text || "").trim().toLowerCase()
  if (!q) return []
  const scored = []
  for (const movie of movies) {
    const title = movie.title.toLowerCase()
    let score = 0
    if (title === q) score = 100
    else if (title.startsWith(q)) score = 80
    else if (title.includes(q)) score = 60
    else if (movie.actors.some(a => a.toLowerCase().includes(q))) score = 40
    else if ((movie.director || "").toLowerCase().includes(q)) score = 35
    else if ((movie.genres || []).some(g => g.toLowerCase().includes(q))) score = 30
    else if (String(movie.year) === q) score = 25
    else if (movie.locations.some(l => l.name.toLowerCase().includes(q))) score = 20
    if (score) scored.push({ score, movie })
  }
  return scored.sort((a, b) => b.score - a.score || a.movie.title.localeCompare(b.movie.title)).slice(0, limit).map(s => s.movie)
}

export function geocodeUrl(text, box) {
  const params = new URLSearchParams({
    q: text,
    format: "jsonv2",
    limit: "5",
    countrycodes: "us",
    bounded: "1",
    viewbox: `${box.west},${box.north},${box.east},${box.south}`
  })
  return `https://nominatim.openstreetmap.org/search?${params}`
}

export function toPlaces(results) {
  const seen = new Set()
  return (results || [])
    .filter(r => !seen.has(r.display_name) && seen.add(r.display_name))
    .map(r => ({ label: r.display_name, lat: Number(r.lat), lng: Number(r.lon) }))
}

export function stats(movies) {
  return {
    movies: movies.length,
    locations: movies.reduce((n, m) => n + m.locations.length, 0),
    posters: movies.filter(m => m.poster).length
  }
}
