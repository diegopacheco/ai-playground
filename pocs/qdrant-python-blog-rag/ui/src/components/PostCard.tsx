import { excerpt, matchPercent, publishedDay } from "../format";
import type { Hit } from "../types";

type Props = {
  hit: Hit;
};

export function PostCard({ hit }: Props) {
  return (
    <article className="card">
      <div className="card-top">
        <a href={hit.url} target="_blank" rel="noreferrer">
          {hit.title}
        </a>
        <span className="score">{matchPercent(hit.score)}</span>
      </div>
      <div className="meta">
        <span>{publishedDay(hit.published)}</span>
        {hit.labels.map((label) => (
          <span key={label} className="label">
            {label}
          </span>
        ))}
      </div>
      <p className="snippet">{excerpt(hit.text, 320)}</p>
    </article>
  );
}
