import { useEffect, useState } from "react";
import { errorMessage, similarVideos } from "../api";
import type { Hit } from "../types";
import { VideoCard } from "./VideoCard";

type Props = {
  video: Hit | null;
  onSimilar: (hit: Hit) => void;
};

export function SimilarPanel({ video, onSimilar }: Props) {
  const [hits, setHits] = useState<Hit[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const videoId = video?.video_id;

  useEffect(() => {
    if (!videoId) return;
    let current = true;
    setLoading(true);
    setError("");
    similarVideos(videoId)
      .then((found) => current && setHits(found))
      .catch((failure) => {
        if (!current) return;
        setHits([]);
        setError(errorMessage(failure));
      })
      .finally(() => current && setLoading(false));
    return () => {
      current = false;
    };
  }, [videoId]);

  if (!video) {
    return <p className="status">Pick "More like this" on any video to see its recommendations.</p>;
  }

  return (
    <section>
      <h2 className="section-title">Selected video</h2>
      <VideoCard hit={{ ...video, score: 0 }} />
      <h2 className="section-title">More like this</h2>
      {loading && <p className="status">Asking Qdrant for recommendations...</p>}
      {error && <p className="status error">{error}</p>}
      {!loading && (
        <div className="results">
          {hits.map((hit) => (
            <VideoCard key={hit.id} hit={hit} onSimilar={onSimilar} />
          ))}
        </div>
      )}
    </section>
  );
}
