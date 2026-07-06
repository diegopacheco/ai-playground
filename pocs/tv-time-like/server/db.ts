import { Database } from "bun:sqlite"
import { mkdirSync } from "node:fs"
import type { AiProvider, CatalogResponse, Episode, LibraryResponse, Media, Metrics, Settings } from "../shared/types.ts"
import { catalog } from "./catalog.ts"

mkdirSync("data", { recursive: true })

export const db = new Database(process.env.DB_PATH || "data/reelmark.db", { create: true })
db.exec("PRAGMA journal_mode = WAL")
db.exec("PRAGMA foreign_keys = ON")
db.exec(`
  CREATE TABLE IF NOT EXISTS media (
    id TEXT PRIMARY KEY,
    provider_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    year INTEGER NOT NULL,
    overview TEXT NOT NULL,
    genres TEXT NOT NULL,
    runtime INTEGER NOT NULL,
    poster TEXT,
    backdrop TEXT,
    color TEXT NOT NULL,
    status TEXT NOT NULL,
    rating REAL NOT NULL,
    in_library INTEGER NOT NULL DEFAULT 0,
    watched INTEGER NOT NULL DEFAULT 0,
    watched_at TEXT,
    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    media_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    number INTEGER NOT NULL,
    title TEXT NOT NULL,
    runtime INTEGER NOT NULL,
    watched INTEGER NOT NULL DEFAULT 0,
    watched_at TEXT,
    FOREIGN KEY(media_id) REFERENCES media(id) ON DELETE CASCADE
  );
  CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS ai_catalog (
    cache_key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    payload TEXT NOT NULL,
    generated_at TEXT NOT NULL
  );
`)
const mediaColumns = db.query("PRAGMA table_info(media)").all() as { name: string }[]
if (!mediaColumns.some(column => column.name === "added_at")) {
  db.exec("ALTER TABLE media ADD COLUMN added_at TEXT")
  db.exec("UPDATE media SET added_at = COALESCE(watched_at, CURRENT_TIMESTAMP) WHERE added_at IS NULL")
}
db.prepare("INSERT OR IGNORE INTO settings (key, value) VALUES ('ai_provider', 'codex')").run()

const addMedia = db.prepare(`
  INSERT OR IGNORE INTO media (id, provider_id, provider, type, title, year, overview, genres, runtime, poster, backdrop, color, status, rating)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
`)
const addEpisode = db.prepare(`
  INSERT OR IGNORE INTO episodes (id, media_id, season, number, title, runtime)
  VALUES (?, ?, ?, ?, ?, ?)
`)

for (const item of catalog) {
  addMedia.run(item.id, item.providerId, item.provider, item.type, item.title, item.year, item.overview, JSON.stringify(item.genres), item.runtime, item.poster, item.backdrop, item.color, item.status, item.rating)
  for (const season of item.seasons || []) {
    season.titles.forEach((title, index) => addEpisode.run(`${item.id}-s${season.season}e${index + 1}`, item.id, season.season, index + 1, title, item.runtime))
  }
}

type MediaRow = Omit<Media, "providerId" | "genres" | "inLibrary" | "watched" | "watchedAt" | "addedAt" | "episodes"> & { provider_id: string; genres: string; in_library: number; watched: number; watched_at: string | null; added_at: string | null }
type EpisodeRow = Omit<Episode, "watched" | "watchedAt"> & { watched: number; watched_at: string | null; media_id: string }

const toEpisode = (row: EpisodeRow): Episode => ({
  id: row.id,
  mediaId: row.media_id,
  season: row.season,
  number: row.number,
  title: row.title,
  runtime: row.runtime,
  watched: Boolean(row.watched),
  watchedAt: row.watched_at
})

const toMedia = (row: MediaRow): Media => ({
  id: row.id,
  providerId: row.provider_id,
  provider: row.provider,
  type: row.type,
  title: row.title,
  year: row.year,
  overview: row.overview,
  genres: JSON.parse(row.genres),
  runtime: row.runtime,
  poster: row.poster,
  backdrop: row.backdrop,
  color: row.color,
  status: row.status,
  rating: row.rating,
  inLibrary: Boolean(row.in_library),
  watched: Boolean(row.watched),
  watchedAt: row.watched_at,
  addedAt: row.added_at || undefined,
  episodes: []
})

export const getMedia = (libraryOnly = false): Media[] => {
  const query = libraryOnly ? "SELECT * FROM media WHERE in_library = 1 ORDER BY watched, title" : "SELECT * FROM media ORDER BY title"
  const media = (db.query(query).all() as MediaRow[]).map(toMedia)
  const episodes = (db.query("SELECT * FROM episodes ORDER BY season, number").all() as EpisodeRow[]).map(toEpisode)
  const byMedia = Map.groupBy(episodes, episode => episode.mediaId)
  return media.map(item => ({ ...item, episodes: byMedia.get(item.id) || [] }))
}

export const libraryTitleKey = (type: string, title: string) => `${type}:${title.trim().toLowerCase()}`

export const libraryTitleKeys = (): Set<string> => new Set(getMedia(true).map(item => libraryTitleKey(item.type, item.title)))

export const getMetrics = (media: Media[]): Metrics => {
  const movies = media.filter(item => item.type === "movie" && item.watched)
  const shows = media.filter(item => item.type === "show")
  const watchedEpisodes = shows.flatMap(item => item.episodes).filter(item => item.watched)
  const genres = new Map<string, number>()
  const months = new Map<string, number>()
  for (const item of [...movies, ...shows.filter(show => show.episodes.some(episode => episode.watched))]) {
    item.genres.forEach(genre => genres.set(genre, (genres.get(genre) || 0) + 1))
  }
  for (const date of [...movies.map(item => item.watchedAt), ...watchedEpisodes.map(item => item.watchedAt)]) {
    if (!date) continue
    const label = new Intl.DateTimeFormat("en", { month: "short" }).format(new Date(date))
    months.set(label, (months.get(label) || 0) + 1)
  }
  const movieMinutes = movies.reduce((sum, item) => sum + item.runtime, 0)
  const showMinutes = watchedEpisodes.reduce((sum, item) => sum + item.runtime, 0)
  const totalMinutes = movieMinutes + showMinutes
  const timestamps = [...movies.map(item => item.watchedAt), ...watchedEpisodes.map(item => item.watchedAt)].filter((date): date is string => Boolean(date)).map(date => new Date(date).getTime()).filter(Number.isFinite)
  const earliest = timestamps.length ? Math.min(...timestamps) : null
  const elapsedWeeks = earliest ? Math.max(1, (Date.now() - earliest) / 604_800_000) : 1
  const allEpisodes = shows.flatMap(item => item.episodes)
  return {
    movieCount: movies.length,
    showCount: shows.filter(show => show.episodes.some(episode => episode.watched)).length,
    episodeCount: watchedEpisodes.length,
    movieMinutes,
    showMinutes,
    totalMinutes,
    averageWeeklyMinutes: Math.round(totalMinutes / elapsedWeeks),
    trackingSince: earliest ? new Date(earliest).toISOString() : null,
    completionRate: allEpisodes.length ? Math.round(watchedEpisodes.length / allEpisodes.length * 100) : 0,
    genreBreakdown: [...genres].map(([genre, count]) => ({ genre, count })).sort((a, b) => b.count - a.count),
    monthlyActivity: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].map(month => ({ month, count: months.get(month) || 0 }))
  }
}

export const getLibrary = (): LibraryResponse => {
  const media = getMedia(true)
  return { media, metrics: getMetrics(media) }
}

export const saveMedia = (media: Media) => {
  db.prepare(`
    INSERT INTO media (id, provider_id, provider, type, title, year, overview, genres, runtime, poster, backdrop, color, status, rating, in_library, added_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
    ON CONFLICT(id) DO UPDATE SET in_library = 1
  `).run(media.id, media.providerId, media.provider, media.type, media.title, media.year, media.overview, JSON.stringify(media.genres), media.runtime, media.poster, media.backdrop, media.color, media.status, media.rating)
  for (const episode of media.episodes) addEpisode.run(episode.id, media.id, episode.season, episode.number, episode.title, episode.runtime)
}

export const getSettings = (): Settings => ({
  aiProvider: (db.query("SELECT value FROM settings WHERE key = 'ai_provider'").get() as { value: AiProvider }).value
})

export const saveSettings = (settings: Settings) => {
  db.prepare("UPDATE settings SET value = ? WHERE key = 'ai_provider'").run(settings.aiProvider)
  return settings
}

export const getCatalogCache = (key: string): CatalogResponse | null => {
  const row = db.query("SELECT payload, generated_at FROM ai_catalog WHERE cache_key = ?").get(key) as { payload: string; generated_at: string } | null
  if (!row || Date.now() - new Date(row.generated_at).getTime() > 21_600_000) return null
  return { ...JSON.parse(row.payload), cached: true }
}

export const saveCatalogCache = (key: string, response: CatalogResponse) => {
  db.prepare("INSERT OR REPLACE INTO ai_catalog (cache_key, provider, payload, generated_at) VALUES (?, ?, ?, ?)").run(key, response.provider, JSON.stringify(response), response.generatedAt)
}
