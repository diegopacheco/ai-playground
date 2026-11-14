import { useCallback, useEffect, useState } from "react";

export const canFullscreen = typeof document !== "undefined" && document.fullscreenEnabled;

export function useFullscreen(): [boolean, () => void] {
  const [full, setFull] = useState(() => canFullscreen && document.fullscreenElement !== null);

  useEffect(() => {
    const sync = () => setFull(document.fullscreenElement !== null);
    document.addEventListener("fullscreenchange", sync);
    return () => document.removeEventListener("fullscreenchange", sync);
  }, []);

  const toggle = useCallback(() => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void document.documentElement.requestFullscreen();
  }, []);

  return [full, toggle];
}
