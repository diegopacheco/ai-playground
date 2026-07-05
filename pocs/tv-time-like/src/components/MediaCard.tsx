import type { Media } from "../../shared/types"
import { Icon } from "./Icon"
import { Poster } from "./Poster"

export function MediaCard({ media, onAdd, busy }: { media: Media; onAdd: (media: Media) => void; busy: boolean }) {
  const watchedEpisodes = media.episodes.filter(item => item.watched).length
  return <article className="media-card">
    <Poster media={media}/>
    <div className="media-card-body">
      <div className="media-kicker"><span>{media.type}</span><span>·</span><span>{media.year || "TBA"}</span>{media.rating > 0 && <><span>·</span><span className="rating"><Icon name="star" size={13}/>{media.rating.toFixed(1)}</span></>}</div>
      <h3>{media.title}</h3>
      <p>{media.overview}</p>
      <div className="genre-row">{media.genres.slice(0, 3).map(genre => <span key={genre}>{genre}</span>)}</div>
      {media.type === "show" && media.episodes.length > 0 && <small className="episode-count">{watchedEpisodes}/{media.episodes.length} episodes watched</small>}
      <button className={media.inLibrary ? "button added" : "button primary"} onClick={() => onAdd(media)} disabled={busy || media.inLibrary}>
        <Icon name={media.inLibrary ? "check" : "plus"} size={17}/>{media.inLibrary ? "In your library" : "Add to library"}
      </button>
    </div>
  </article>
}
