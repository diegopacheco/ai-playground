import type { Media } from "../../shared/types"
import { EpisodeList } from "./EpisodeList"
import { Icon } from "./Icon"
import { Poster } from "./Poster"
import { ProgressRing } from "./ProgressRing"
import { useEffect, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { api } from "../api/client"

export function LibraryDetailDialog({ media, onClose, onMediaToggle, onEpisodeToggle, onRemove, busy }: { media: Media | null; onClose: () => void; onMediaToggle: (id: string) => void; onEpisodeToggle: (id: string) => void; onRemove: (id: string) => void; busy: boolean }) {
  const [selectedSeason, setSelectedSeason] = useState(1)
  useEffect(() => {
    if (media?.episodes.length) setSelectedSeason(Math.max(...media.episodes.map(episode => episode.season)))
  }, [media?.id])
  useEffect(() => {
    if (!media) return
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose()
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [media, onClose])
  const searchTerms = media ? `${media.title} ${media.year || ""} official trailer` : ""
  const trailer = useQuery({ queryKey: ["trailer", media?.id], queryFn: () => api.trailer(searchTerms), enabled: Boolean(media), staleTime: Infinity })
  if (!media) return null
  const watched = media.episodes.filter(item => item.watched).length
  const progress = media.type === "movie" ? Number(media.watched) * 100 : Math.round(watched / Math.max(media.episodes.length, 1) * 100)
  const seasons = new Map([...Map.groupBy(media.episodes, episode => episode.season)].sort((a, b) => b[0] - a[0]))
  const selectedEpisodes = seasons.get(selectedSeason) || []
  const trailerUrl = `https://www.youtube.com/results?search_query=${encodeURIComponent(searchTerms)}`
  return <div className="dialog-backdrop detail-backdrop" role="presentation" onMouseDown={event => event.target === event.currentTarget && onClose()}>
    <section className="library-detail" role="dialog" aria-modal="true" aria-labelledby="detail-title">
      <button className="dialog-close" onClick={onClose} aria-label="Close"><Icon name="close"/></button>
      <header className="detail-header"><Poster media={media}/><div className="detail-copy"><span className="eyebrow">{media.type} · {media.year || media.status}</span><h2 id="detail-title">{media.title}</h2><p>{media.overview}</p><div className="genre-row">{media.genres.map(genre => <span key={genre}>{genre}</span>)}</div><div className="library-actions">{media.type === "movie" && <button className={media.watched ? "button added" : "button primary"} onClick={() => onMediaToggle(media.id)} disabled={busy}><Icon name="check" size={17}/>{media.watched ? "Watched" : "Mark watched"}</button>}<button className="button secondary" onClick={() => { onRemove(media.id); onClose() }} disabled={busy}><Icon name="trash" size={16}/>Remove</button></div></div><ProgressRing value={progress} label={media.type === "movie" ? "complete" : `${watched}/${media.episodes.length}`}/></header>
      <div className="detail-trailer">{trailer.data?.videoId ? <iframe src={`https://www.youtube.com/embed/${trailer.data.videoId}?rel=0`} title={`${media.title} trailer`} allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen loading="lazy"/> : <a className="trailer-fallback" href={trailerUrl} target="_blank" rel="noreferrer"><Icon name="play" size={22}/><span>{trailer.isFetching ? "Finding the trailer…" : "Open trailer search on YouTube"}</span></a>}</div>
      {media.type === "show" && <div className="detail-seasons"><div className="season-picker"><span>Choose season</span><div>{[...seasons].map(([season, episodes]) => <button key={season} className={selectedSeason === season ? "active" : ""} onClick={() => setSelectedSeason(season)}><strong>{season}</strong><small>{episodes.filter(item => item.watched).length}/{episodes.length}</small></button>)}</div></div><section className="season-block"><div className="season-heading"><div><small>Season</small><strong>{String(selectedSeason).padStart(2, "0")}</strong></div><span>{selectedEpisodes.filter(item => item.watched).length} of {selectedEpisodes.length} watched</span></div><EpisodeList episodes={selectedEpisodes} onToggle={onEpisodeToggle} busy={busy}/></section></div>}
    </section>
  </div>
}
