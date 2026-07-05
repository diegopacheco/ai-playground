import type { Media } from "../shared/types.ts"
import { getMedia } from "./db.ts"

const palette = ["#e86d52", "#2f6d64", "#d9947c", "#3f7890", "#a55f35", "#496c4f"]

type TvMazeResult = {
  show: {
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
  }
}

type TvMazeEpisode = {
  id: number
  season: number
  number: number
  name: string
  runtime: number | null
}

const stripTags = (value: string | null) => value?.replace(/<[^>]+>/g, "") || "No synopsis is available yet."

export const searchMedia = async (query: string): Promise<Media[]> => {
  const local = getMedia().filter(item => item.title.toLowerCase().includes(query.toLowerCase()))
  if (!query.trim()) return local.filter(item => item.provider === "local").slice(0, 8)
  const shows = await fetch(`https://api.tvmaze.com/search/shows?q=${encodeURIComponent(query)}`)
    .then(response => response.ok ? response.json() as Promise<TvMazeResult[]> : [])
    .catch(() => [])
  const remoteShows: Media[] = await Promise.all(shows.slice(0, 6).map(async ({ show }, index) => {
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
  }))
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
  return [...local, ...remoteShows.filter(item => !known.has(item.id)), ...movies].slice(0, 14)
}
