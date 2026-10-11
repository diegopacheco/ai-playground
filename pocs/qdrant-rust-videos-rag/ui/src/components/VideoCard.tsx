import type { Hit } from "../types";

type Props = {
  hit: Hit;
  onSimilar?: (hit: Hit) => void;
};

export function VideoCard({ hit, onSimilar }: Props) {
  return (
    <article className="card video">
      <a className="thumbnail" href={hit.url} target="_blank" rel="noreferrer">
        <img src={hit.thumbnail} alt="" loading="lazy" />
        {hit.duration_text && <span className="duration">{hit.duration_text}</span>}
      </a>
      <div className="video-body">
        <a className="video-title" href={hit.url} target="_blank" rel="noreferrer">
          {hit.title}
        </a>
        <p className="meta">
          <span>{hit.views.toLocaleString("en-US")} views</span>
          {hit.age_text && <span>{hit.age_text}</span>}
          {hit.score > 0 && <span className="score">score {hit.score.toFixed(3)}</span>}
        </p>
        {hit.description && <p className="description">{hit.description}</p>}
        {onSimilar && (
          <button className="secondary" onClick={() => onSimilar(hit)}>
            More like this
          </button>
        )}
      </div>
    </article>
  );
}
