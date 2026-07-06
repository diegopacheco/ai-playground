import { useEffect, useState } from "react"
import type { Media } from "../../shared/types"
import { Icon } from "./Icon"

export function Poster({ media, className = "" }: { media: Media; className?: string }) {
  const [failed, setFailed] = useState(false)
  useEffect(() => setFailed(false), [media.poster])
  if (media.poster && !failed) return <div className={`poster poster-image ${className}`} style={{ "--poster-color": media.color } as React.CSSProperties}><img src={media.poster} alt={`${media.title} poster`} loading="eager" onError={() => setFailed(true)}/></div>
  const words = media.title.split(/[ :]/).filter(Boolean)
  const monogram = words.slice(0, 2).map(word => word[0]).join("")
  return <div className={`poster poster-fallback ${className}`} style={{ "--poster-color": media.color } as React.CSSProperties}>
    <span className="poster-type"><Icon name={media.type === "movie" ? "film" : "tv"} size={14}/>{media.type}</span>
    <strong>{monogram}</strong>
    <div><b>{media.title}</b><small>{media.year}</small></div>
  </div>
}
