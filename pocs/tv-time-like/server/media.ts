import type { Media } from "../shared/types.ts"
import { getMedia, libraryTitleKey, libraryTitleKeys } from "./db.ts"

const palette = ["#e86d52", "#2f6d64", "#d9947c", "#3f7890", "#a55f35", "#496c4f"]

type TvMazeShow = {
  id: number
  name: string
  premiered: string | null
  summary: string | null
  genres: string[]
  runtime: number | null
  averageRuntime: number | null
  image: { medium: string; original: string } | null
  status: string
  rating: { average: number | null }
  weight?: number
}

type TvMazeResult = { show: TvMazeShow }

type TvMazeEpisode = {
  id: number
  season: number
  number: number
  name: string
  runtime: number | null
}

const stripTags = (value: string | null) => value?.replace(/<[^>]+>/g, "") || "No synopsis is available yet."

const buildShow = async (show: TvMazeShow, index: number): Promise<Media> => {
  const episodes = await fetch(`https://api.tvmaze.com/shows/${show.id}/episodes`)
    .then(response => response.ok ? response.json() as Promise<TvMazeEpisode[]> : [])
    .catch(() => [])
  return {
    id: `tvmaze-${show.id}`,
    providerId: String(show.id),
    provider: "tvmaze",
    type: "show",
    title: show.name,
    year: Number(show.premiered?.slice(0, 4)) || 0,
    overview: stripTags(show.summary),
    genres: show.genres,
    runtime: show.averageRuntime || show.runtime || 45,
    poster: show.image?.original || show.image?.medium || null,
    backdrop: null,
    color: palette[index % palette.length],
    status: show.status,
    rating: show.rating.average || 0,
    inLibrary: false,
    watched: false,
    watchedAt: null,
    episodes: episodes.map(episode => ({
      id: `tvmaze-episode-${episode.id}`,
      mediaId: `tvmaze-${show.id}`,
      season: episode.season,
      number: episode.number,
      title: episode.name,
      runtime: episode.runtime || show.averageRuntime || 45,
      watched: false,
      watchedAt: null
    }))
  }
}

export const searchMedia = async (query: string): Promise<Media[]> => {
  const owned = libraryTitleKeys()
  const available = (item: { type: string; title: string }) => !owned.has(libraryTitleKey(item.type, item.title))
  const local = getMedia().filter(item => item.title.toLowerCase().includes(query.toLowerCase()))
  if (!query.trim()) {
    const movies = local.filter(item => item.provider === "local" && item.type === "movie" && available(item)).slice(0, 6)
    const date = new Date().toISOString().slice(0, 10)
    const schedule = await fetch(`https://api.tvmaze.com/schedule/web?date=${date}`)
      .then(response => response.ok ? response.json() as Promise<{ _embedded?: { show?: TvMazeShow } }[]> : [])
      .catch(() => [])
    const unique = new Map<number, TvMazeShow>()
    for (const entry of schedule) {
      const show = entry._embedded?.show
      if (show?.name && available({ type: "show", title: show.name })) unique.set(show.id, show)
    }
    const ranked = [...unique.values()].sort((a, b) => (b.weight || 0) - (a.weight || 0)).slice(0, 6)
    const shows = await Promise.all(ranked.map((show, index) => buildShow(show, index)))
    return [...shows, ...movies]
  }
  const shows = await fetch(`https://api.tvmaze.com/search/shows?q=${encodeURIComponent(query)}`)
    .then(response => response.ok ? response.json() as Promise<TvMazeResult[]> : [])
    .catch(() => [])
  const remoteShows = await Promise.all(shows.slice(0, 6).map(({ show }, index) => buildShow(show, index)))
  const token = process.env.TMDB_ACCESS_TOKEN
  let movies: Media[] = []
  if (token) {
    type TmdbMovie = { id: number; title: string; release_date: string; overview: string; genre_ids: number[]; poster_path: string | null; backdrop_path: string | null; vote_average: number }
    const response = await fetch(`https://api.themoviedb.org/3/search/movie?query=${encodeURIComponent(query)}&include_adult=false`, { headers: { Authorization: `Bearer ${token}` } }).catch(() => null)
    const data = response?.ok ? await response.json() as { results: TmdbMovie[] } : { results: [] }
    movies = data.results.slice(0, 6).map((movie, index) => ({
      id: `tmdb-${movie.id}`,
      providerId: String(movie.id),
      provider: "tmdb",
      type: "movie",
      title: movie.title,
      year: Number(movie.release_date?.slice(0, 4)) || 0,
      overview: movie.overview || "No synopsis is available yet.",
      genres: [],
      runtime: 0,
      poster: movie.poster_path ? `https://image.tmdb.org/t/p/w780${movie.poster_path}` : null,
      backdrop: movie.backdrop_path ? `https://image.tmdb.org/t/p/w1280${movie.backdrop_path}` : null,
      color: palette[(index + 2) % palette.length],
      status: "Released",
      rating: movie.vote_average,
      inLibrary: false,
      watched: false,
      watchedAt: null,
      episodes: []
    }))
  }
  const known = new Set(local.map(item => item.id))
  return [...local, ...remoteShows.filter(item => !known.has(item.id)), ...movies].filter(available).slice(0, 14)
}
