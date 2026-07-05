import { useState } from "react"
import type { Media } from "../../shared/types"
import { EpisodeList } from "./EpisodeList"
import { Icon } from "./Icon"
import { Poster } from "./Poster"
import { ProgressRing } from "./ProgressRing"

export function LibraryItem({ media, onMediaToggle, onEpisodeToggle, onRemove, busy }: { media: Media; onMediaToggle: (id: string) => void; onEpisodeToggle: (id: string) => void; onRemove: (id: string) => void; busy: boolean }) {
  const [expanded, setExpanded] = useState(false)
  const watched = media.episodes.filter(item => item.watched).length
  const progress = media.type === "movie" ? Number(media.watched) * 100 : Math.round(watched / Math.max(media.episodes.length, 1) * 100)
  const seasons = Map.groupBy(media.episodes, episode => episode.season)
  return <article className={expanded ? "library-item expanded" : "library-item"}>
    <div className="library-summary">
      <Poster media={media}/>
      <div className="library-copy">
        <div className="media-kicker"><span>{media.type}</span><span>·</span><span>{media.year}</span><span>·</span><span>{media.status}</span></div>
        <h3>{media.title}</h3>
        <p>{media.overview}</p>
        <div className="genre-row">{media.genres.map(genre => <span key={genre}>{genre}</span>)}</div>
        <div className="library-actions">
          {media.type === "movie" ? <button className={media.watched ? "button added" : "button primary"} onClick={() => onMediaToggle(media.id)} disabled={busy}><Icon name="check" size={17}/>{media.watched ? "Watched" : "Mark watched"}</button> : <button className="button secondary" onClick={() => setExpanded(value => !value)}><Icon name="play" size={16}/>{expanded ? "Hide episodes" : "View episodes"}</button>}
          <button className="icon-button danger" onClick={() => onRemove(media.id)} disabled={busy} aria-label={`Remove ${media.title}`}><Icon name="trash" size={17}/></button>
        </div>
      </div>
      <ProgressRing value={progress} label={media.type === "movie" ? "complete" : `${watched}/${media.episodes.length}`}/>
    </div>
    {expanded && media.type === "show" && <div className="season-stack">
      {[...seasons].map(([season, episodes]) => <section key={season} className="season-block">
        <div className="season-heading"><div><small>Season</small><strong>{String(season).padStart(2, "0")}</strong></div><span>{episodes.filter(item => item.watched).length} of {episodes.length} watched</span></div>
        <EpisodeList episodes={episodes} onToggle={onEpisodeToggle} busy={busy}/>
      </section>)}
    </div>}
  </article>
}
