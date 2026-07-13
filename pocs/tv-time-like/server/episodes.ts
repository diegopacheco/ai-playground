import { db, getMedia } from "./db.ts"
import type { Media } from "../shared/types.ts"

type TvMazeEpisode = {
  id: number
  season: number
  number: number
  name: string
  runtime: number | null
}

type TvMazeShow = {
  id: number
  name: string
  averageRuntime: number | null
  runtime: number | null
  _embedded?: { episodes?: TvMazeEpisode[] }
}

export type EpisodeSyncReport = {
  showsScanned: number
  showsMatched: number
  episodesAdded: number
  episodesUpdated: number
  unmatched: string[]
  failures: string[]
}

db.exec(`
  CREATE TABLE IF NOT EXISTS tvmaze_matches (
    media_id TEXT PRIMARY KEY,
    tvmaze_id TEXT NOT NULL,
    FOREIGN KEY(media_id) REFERENCES media(id) ON DELETE CASCADE
  )
`)

const matchByMedia = db.prepare("SELECT tvmaze_id FROM tvmaze_matches WHERE media_id = ?")
const saveMatch = db.prepare("INSERT OR REPLACE INTO tvmaze_matches (media_id, tvmaze_id) VALUES (?, ?)")
const findEpisode = db.prepare("SELECT id, title, runtime FROM episodes WHERE media_id = ? AND season = ? AND number = ?")
const addEpisode = db.prepare("INSERT INTO episodes (id, media_id, season, number, title, runtime) VALUES (?, ?, ?, ?, ?, ?)")
const updateEpisode = db.prepare("UPDATE episodes SET title = ?, runtime = ? WHERE id = ?")
const legacyEpisodes = db.prepare("SELECT id, season, watched, watched_at FROM episodes WHERE media_id = ? AND id LIKE 'tvtime-episode-%' AND number > 1000 ORDER BY season, CAST(REPLACE(id, 'tvtime-episode-', '') AS INTEGER)")
const transferWatchHistory = db.prepare("UPDATE episodes SET watched = MAX(watched, ?), watched_at = COALESCE(watched_at, ?) WHERE media_id = ? AND season = ? AND number = ?")
const removeEpisode = db.prepare("DELETE FROM episodes WHERE id = ?")

const delay = (milliseconds: number) => new Promise(resolve => setTimeout(resolve, milliseconds))
const normalized = (value: string) => value.toLowerCase().replace(/[^a-z0-9]/g, "")

const fetchTvMaze = async <T>(url: string): Promise<T> => {
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const response = await fetch(url)
    if (response.ok) return response.json() as Promise<T>
    if (response.status === 404) return null as T
    if (response.status !== 429) throw new Error(`TVmaze returned ${response.status}`)
    await delay(1_000)
  }
  throw new Error("TVmaze rate limit did not clear")
}

const cachedTvMazeId = (media: Media) => {
  if (media.provider === "tvmaze") return media.providerId
  return (matchByMedia.get(media.id) as { tvmaze_id: string } | null)?.tvmaze_id || null
}

const loadShow = async (media: Media): Promise<TvMazeShow | null> => {
  const knownId = cachedTvMazeId(media)
  if (knownId) {
    const show = await fetchTvMaze<TvMazeShow>(`https://api.tvmaze.com/shows/${knownId}?embed=episodes`)
    if (!show) return null
    saveMatch.run(media.id, String(show.id))
    return show
  }
  const show = await fetchTvMaze<TvMazeShow | null>(`https://api.tvmaze.com/singlesearch/shows?q=${encodeURIComponent(media.title)}&embed=episodes`)
  if (!show || normalized(show.name) !== normalized(media.title)) return null
  saveMatch.run(media.id, String(show.id))
  return show
}

const saveEpisodes = (media: Media, show: TvMazeShow) => {
  let episodesAdded = 0
  let episodesUpdated = 0
  for (const episode of show._embedded?.episodes || []) {
    const existing = findEpisode.get(media.id, episode.season, episode.number) as { id: string; title: string; runtime: number } | null
    const runtime = episode.runtime || show.averageRuntime || show.runtime || media.runtime || 45
    if (!existing) {
      addEpisode.run(`tvmaze-episode-${episode.id}`, media.id, episode.season, episode.number, episode.name, runtime)
      episodesAdded += 1
    } else if (existing.title !== episode.name || existing.runtime !== runtime) {
      updateEpisode.run(episode.name, runtime, existing.id)
      episodesUpdated += 1
    }
  }
  const remoteBySeason = Map.groupBy(show._embedded?.episodes || [], episode => episode.season)
  const legacyBySeason = Map.groupBy(legacyEpisodes.all(media.id) as { id: string; season: number; watched: number; watched_at: string | null }[], episode => episode.season)
  for (const [season, imported] of legacyBySeason) {
    const canonical = (remoteBySeason.get(season) || []).toSorted((a, b) => a.number - b.number)
    imported.forEach((episode, index) => {
      const match = canonical[index]
      if (!match) return
      transferWatchHistory.run(episode.watched, episode.watched_at, media.id, season, match.number)
      removeEpisode.run(episode.id)
      episodesUpdated += 1
    })
  }
  return { episodesAdded, episodesUpdated }
}

const emptyReport = (): EpisodeSyncReport => ({
  showsScanned: 0,
  showsMatched: 0,
  episodesAdded: 0,
  episodesUpdated: 0,
  unmatched: [],
  failures: []
})

const syncShow = async (media: Media, report: EpisodeSyncReport) => {
  report.showsScanned += 1
  try {
    const show = await loadShow(media)
    if (!show) {
      report.unmatched.push(media.title)
      return
    }
    report.showsMatched += 1
    const saved = saveEpisodes(media, show)
    report.episodesAdded += saved.episodesAdded
    report.episodesUpdated += saved.episodesUpdated
  } catch (error) {
    report.failures.push(`${media.title}: ${error instanceof Error ? error.message : "sync failed"}`)
  }
}

export const syncMediaEpisodes = async (mediaId: string): Promise<EpisodeSyncReport> => {
  const report = emptyReport()
  const media = getMedia(true).find(item => item.id === mediaId && item.type === "show")
  if (!media) return report
  await syncShow(media, report)
  return report
}

export const syncLibraryEpisodes = async (): Promise<EpisodeSyncReport> => {
  const report = emptyReport()
  const shows = getMedia(true).filter(item => item.type === "show")
  const worker = async () => {
    while (shows.length) {
      const show = shows.shift()
      if (show) await syncShow(show, report)
      await delay(1_000)
    }
  }
  await Promise.all([worker(), worker()])
  return report
}
