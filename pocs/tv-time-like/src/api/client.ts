import type { CatalogResponse, ImportReport, LibraryResponse, Media, Settings } from "../../shared/types"

const request = async <T>(url: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(url, init)
  if (!response.ok) throw new Error((await response.json().catch(() => null))?.error || "Request failed")
  return response.json()
}

export const api = {
  library: () => request<LibraryResponse>("/api/library"),
  search: (query: string) => request<Media[]>(`/api/search?q=${encodeURIComponent(query)}`),
  add: (media: Media) => request<Media>("/api/media", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(media) }),
  remove: (id: string) => request<{ removed: boolean }>(`/api/library/${id}`, { method: "DELETE" }),
  toggleMedia: (id: string) => request<{ watched: boolean }>(`/api/media/${id}/watched`, { method: "PATCH" }),
  toggleEpisode: (id: string) => request<{ watched: boolean }>(`/api/episodes/${id}/watched`, { method: "PATCH" }),
  importTvTime: (text: string) => request<ImportReport>("/api/import/tvtime", { method: "POST", headers: { "Content-Type": "text/csv" }, body: text }),
  settings: () => request<Settings>("/api/settings"),
  saveSettings: (settings: Settings) => request<Settings>("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(settings) }),
  aiCatalog: (topic: string, refresh = false) => request<CatalogResponse>("/api/catalog/ai", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ topic, refresh }) }),
  syncMetadata: () => request<{ scanned: number; imagesUpdated: number; genresUpdated: number; missingImages: number; pagesResolved: number }>("/api/metadata/sync", { method: "POST" })
}
