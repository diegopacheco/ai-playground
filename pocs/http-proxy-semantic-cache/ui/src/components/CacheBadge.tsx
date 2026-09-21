export function CacheBadge({ cached }: { cached: boolean }) {
  return <span className={cached ? "badge hit" : "badge miss"}>{cached ? "CACHE HIT" : "CACHE MISS"}</span>;
}
