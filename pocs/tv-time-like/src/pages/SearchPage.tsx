import { useQuery } from "@tanstack/react-query"
import { useMemo, useState } from "react"
import { api } from "../api/client"
import { Icon } from "../components/Icon"
import { MediaCard } from "../components/MediaCard"
import { AiCatalog } from "../components/AiCatalog"
import { useLibraryActions } from "../hooks/useLibrary"

type Filter = "all" | "movie" | "show"

export function SearchPage() {
  const [query, setQuery] = useState("")
  const [filter, setFilter] = useState<Filter>("all")
  const search = useQuery({ queryKey: ["search", query], queryFn: () => api.search(query), staleTime: 300_000 })
  const actions = useLibraryActions()
  const results = useMemo(() => (search.data || []).filter(item => filter === "all" || item.type === filter), [search.data, filter])
  return <div className="page discover-page">
    <section className="hero">
      <div className="hero-copy"><span className="eyebrow">Your personal watch journal</span><h1>Find it. Watch it.<br/><em>Remember it.</em></h1><p>A quiet place for every film, every season and every late-night episode.</p></div>
      <div className="hero-stamp"><span>Since</span><strong>2026</strong><small>Locally kept</small></div>
    </section>
    <section className="search-panel">
      <label className="search-box"><Icon name="search" size={23}/><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Search for a movie or television show…" autoFocus/><kbd>⌘ K</kbd></label>
      <div className="filter-row"><div className="segmented" aria-label="Filter results">{(["all", "movie", "show"] as Filter[]).map(value => <button key={value} className={filter === value ? "active" : ""} onClick={() => setFilter(value)}>{value === "all" ? "All titles" : value === "movie" ? "Movies" : "TV shows"}</button>)}</div><span>{results.length} titles</span></div>
    </section>
    <section className="results-section">
      <div className="section-title"><div><span className="eyebrow">{query ? "Search results" : "Start your collection"}</span><h2>{query ? `Titles matching “${query}”` : "A few worthy first picks"}</h2></div><span className="source-note">Live show data from TVmaze</span></div>
      {search.isLoading ? <div className="loading-grid">{Array.from({ length: 6 }, (_, index) => <div className="loading-card" key={index}/>)}</div> : <div className="media-grid">{results.map(media => <MediaCard key={media.id} media={media} onAdd={item => actions.add.mutate(item)} busy={actions.add.isPending}/>)}</div>}
      {!search.isLoading && !results.length && <div className="empty-state"><span>00</span><h3>No titles found</h3><p>Try a shorter title or switch the media filter.</p></div>}
    </section>
    <AiCatalog onAdd={item => actions.add.mutate(item)}/>
  </div>
}
