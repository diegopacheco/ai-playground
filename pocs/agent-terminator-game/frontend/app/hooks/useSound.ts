import { useCallback, useRef, useState } from "react";

const SOUNDS = {
  kill: "/sounds/hasta-la-vista.mp3",
  date: "/sounds/mosquito-date.mp3",
  hatch: "/sounds/egg-hatch.mp3",
  footstep: "/sounds/footstep.mp3",
  start: "/sounds/ill-be-back.mp3",
  victory: "/sounds/victory.mp3",
  swarm: "/sounds/swarm.mp3",
} as const;

type SoundName = keyof typeof SOUNDS;

export function useSound() {
  const [muted, setMuted] = useState(false);
  const audioCache = useRef<Map<string, HTMLAudioElement>>(new Map());

  const play = useCallback(
    (name: SoundName) => {
      if (muted) return;
      try {
        let audio = audioCache.current.get(name);
        if (!audio) {
          audio = new Audio(SOUNDS[name]);
          audioCache.current.set(name, audio);
        }
        audio.currentTime = 0;
        audio.volume = 0.5;
        audio.play().catch(() => {});
      } catch {}
    },
    [muted]
  );

  const toggleMute = useCallback(() => {
    setMuted((prev) => !prev);
  }, []);

  return { play, muted, toggleMute };
}
