import type { ImportReport } from "../shared/types.ts"
import { db } from "./db.ts"

export const parseCsv = (text: string): Record<string, string>[] => {
  const rows: string[][] = []
  let row: string[] = []
  let field = ""
  let quoted = false
  for (let index = 0; index < text.length; index++) {
    const char = text[index]
    if (char === '"' && quoted && text[index + 1] === '"') {
      field += '"'
      index++
    } else if (char === '"') quoted = !quoted
    else if (char === "," && !quoted) {
      row.push(field)
      field = ""
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && text[index + 1] === "\n") index++
      row.push(field)
      if (row.some(value => value.trim())) rows.push(row)
      row = []
      field = ""
    } else field += char
  }
  if (field || row.length) {
    row.push(field)
    rows.push(row)
  }
  const headers = rows.shift()?.map(value => value.trim().toLowerCase()) || []
  return rows.map(values => Object.fromEntries(headers.map((header, index) => [header, values[index]?.trim() || ""])))
}

const findValue = (row: Record<string, string>, names: string[]) => {
  const key = Object.keys(row).find(header => names.some(name => header.includes(name)))
  return key ? row[key] : ""
}

const minutes = (value: string, fallback: number) => {
  const runtime = Number(value)
  if (!runtime) return fallback
  return runtime > 300 ? Math.round(runtime / 60) : runtime
}

export const importTvTime = async (text: string): Promise<ImportReport> => {
  const rows = parseCsv(text)
  let imported = 0
  let skipped = 0
  const messages: string[] = []
  const mediaStatement = db.prepare(`
    INSERT INTO media (id, provider_id, provider, type, title, year, overview, genres, runtime, poster, backdrop, color, status, rating, in_library, added_at)
    VALUES (?, ?, 'tvtime', 'show', ?, 0, 'Imported from TV Time', '[]', ?, NULL, NULL, ?, 'Imported', 0, 1, CURRENT_TIMESTAMP)
    ON CONFLICT(id) DO UPDATE SET title = excluded.title, runtime = excluded.runtime, in_library = 1
  `)
  const episodeStatement = db.prepare(`
    INSERT INTO episodes (id, media_id, season, number, title, runtime, watched, watched_at)
    VALUES (?, ?, ?, ?, ?, ?, 1, ?)
    ON CONFLICT(id) DO UPDATE SET watched = 1, watched_at = COALESCE(episodes.watched_at, excluded.watched_at)
  `)
  const movieStatement = db.prepare(`
    INSERT INTO media (id, provider_id, provider, type, title, year, overview, genres, runtime, poster, backdrop, color, status, rating, in_library, watched, watched_at, added_at)
    VALUES (?, ?, 'tvtime', 'movie', ?, ?, 'Imported from TV Time', '[]', ?, NULL, NULL, ?, 'Released', 0, 1, ?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(id) DO UPDATE SET in_library = 1, watched = MAX(media.watched, excluded.watched), watched_at = COALESCE(media.watched_at, excluded.watched_at)
  `)
  const mediaIds = new Set<string>()
  const episodeIds = new Set<string>()
  const movieIds = new Set<string>()
  const colors = ["#e86d52", "#2f6d64", "#d9947c", "#3f7890", "#a55f35", "#496c4f"]
  const legacy = rows.length > 0 && Object.hasOwn(rows[0], "movie_name")
  const write = db.transaction(() => {
    for (const row of rows) {
      if (legacy) {
        const title = row.movie_name
        const providerId = row.uuid
        if (!title || !providerId) continue
        const id = `tvtime-movie-${providerId}`
        const watched = ["watch", "rewatch", "rewatch_count"].includes(row.type)
        const color = colors[[...title].reduce((sum, char) => sum + char.charCodeAt(0), 0) % colors.length]
        movieStatement.run(id, providerId, title, Number(row.release_date?.slice(0, 4)) || 0, minutes(row.runtime, 0), color, watched ? 1 : 0, watched ? row.created_at || new Date().toISOString() : null)
        movieIds.add(id)
        if (watched) imported++
        continue
      }
      const title = findValue(row, ["series_name", "show_name", "title", "name"])
      const providerId = row.s_id || title.toLowerCase().replace(/[^a-z0-9]+/g, "-")
      if (!title || !providerId) {
        skipped++
        continue
      }
      const mediaId = `tvtime-${providerId}`
      if (!mediaIds.has(mediaId)) {
        const color = colors[[...title].reduce((sum, char) => sum + char.charCodeAt(0), 0) % colors.length]
        mediaStatement.run(mediaId, providerId, title, minutes(row.runtime, 45), color)
        mediaIds.add(mediaId)
      }
      const season = Number(findValue(row, ["season_number", "season"]))
      const number = Number(findValue(row, ["episode_number", "episode"]))
      const providerEpisodeId = row.ep_id || row.episode_id
      if (!season || !number || !providerEpisodeId) {
        skipped++
        continue
      }
      const id = `tvtime-episode-${providerEpisodeId}`
      if (episodeIds.has(id)) continue
      episodeStatement.run(id, mediaId, season, number, `Episode ${number}`, minutes(row.runtime, 45), row.updated_at || row.created_at || new Date().toISOString())
      episodeIds.add(id)
      imported++
    }
  })
  write()
  if (legacy) messages.push(`${movieIds.size} movies added to your library`)
  else messages.push(`${mediaIds.size} shows added to your library`)
  return { imported, skipped, messages }
}
