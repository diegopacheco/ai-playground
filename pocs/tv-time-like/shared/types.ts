export type MediaType = "movie" | "show"

export type Episode = {
  id: string
  mediaId: string
  season: number
  number: number
  title: string
  runtime: number
  watched: boolean
  watchedAt: string | null
}

export type Media = {
  id: string
  providerId: string
  provider: string
  type: MediaType
  title: string
  year: number
  overview: string
  genres: string[]
  runtime: number
  poster: string | null
  backdrop: string | null
  color: string
  status: string
  rating: number
  inLibrary: boolean
  watched: boolean
  watchedAt: string | null
  addedAt?: string
  episodes: Episode[]
}

export type Metrics = {
  movieCount: number
  showCount: number
  episodeCount: number
  movieMinutes: number
  showMinutes: number
  totalMinutes: number
  averageWeeklyMinutes: number
  trackingSince: string | null
  completionRate: number
  genreBreakdown: { genre: string; count: number }[]
  monthlyActivity: { month: string; count: number }[]
}

export type LibraryResponse = {
  media: Media[]
  metrics: Metrics
}

export type ImportReport = {
  imported: number
  skipped: number
  messages: string[]
}

export type AiProvider = "codex" | "claude" | "gemini"

export type CatalogItem = {
  media: Media
  sourceUrl: string
  sourceName: string
  reason: string
}

export type CatalogResponse = {
  provider: AiProvider
  cached: boolean
  generatedAt: string
  items: CatalogItem[]
}

export type Settings = {
  aiProvider: AiProvider
}
