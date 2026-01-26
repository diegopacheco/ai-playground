import React, { useEffect, useRef, useState } from "react";

interface SongProps {
  song: {
    title: string;
    artist: string;
    audioUrl: string;
    lyrics: string[];
  };
}

const Song: React.FC<SongProps> = ({ song }) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [currentLine, setCurrentLine] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    setCurrentLine(0);
    setPlaying(false);
  }, [song]);

  useEffect(() => {
    if (!playing) return;
    const interval = setInterval(() => {
      setCurrentLine((prev) => {
        const next = prev + 1;
        if (next >= song.lyrics.length) {
          clearInterval(interval);
          audioRef.current?.pause();
          setPlaying(false);
          return prev;
        }
        return next;
      });
    }, 3000);
    return () => clearInterval(interval);
  }, [playing, song.lyrics.length]);

  const handlePlay = () => {
    audioRef.current?.play();
    setPlaying(true);
  };
  const handlePause = () => {
    audioRef.current?.pause();
    setPlaying(false);
  };
  const handleStop = () => {
    audioRef.current?.pause();
    audioRef.current!.currentTime = 0;
    setPlaying(false);
    setCurrentLine(0);
  };

  return (
    <div style={{ marginTop: "20px" }}>
      <h2>{song.title} – {song.artist}</h2>
      <div style={{ marginBottom: "10px" }}>
        {song.lyrics.map((line, i) => (
          <div key={i} style={{ color: i === currentLine ? "red" : "black" }}>
            {line}
          </div>
        ))}
      </div>
      <audio ref={audioRef} src={song.audioUrl} /\u003e
      <div>
        <button onClick={handlePlay}>Play</button>
        <button onClick={handlePause}>Pause</button>
        <button onClick={handleStop}>Stop</button>
      </div>
    </div>
  );
};

export default Song;