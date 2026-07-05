import type { Media } from "../../shared/types"
import { Poster } from "./Poster"

export function LibraryGridCard({ media, onOpen }: { media: Media; onOpen: () => void }) {
  const watched = media.episodes.filter(item => item.watched).length
  const progress = media.type === "movie" ? Number(media.watched) * 100 : Math.round(watched / Math.max(media.episodes.length, 1) * 100)
  return <article className="library-grid-card">
    <button className="cover-button" onClick={onOpen} aria-label={`Open ${media.title}`}><Poster media={media}/><span className="cover-summary">{media.overview}</span><span className="cover-type">{media.type === "movie" ? "Movie" : `${media.episodes.length} episodes`}</span></button>
    <button className="cover-copy" onClick={onOpen}><strong>{media.title}</strong><span>{media.year || media.status}</span></button>
    <div className="cover-progress"><div><i style={{ width: `${progress}%` }}/></div><span>{progress}%</span></div>
  </article>
}
