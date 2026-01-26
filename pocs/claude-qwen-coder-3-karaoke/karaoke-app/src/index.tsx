import React, { useState, useEffect, useRef } from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import './styles.css';

// Mock song data
const mockSongs = [
  {
    id: 1,
    title: "Bohemian Rhapsody",
    artist: "Queen",
    lyrics: [
      "Is this real life? Is this just fantasy?",
      "Caught in a landslide",
      "No escape from reality",
      "Open your eyes",
      "Look up to the skies and see",
      "I'm just a poor boy, nobody loves me",
      "He's just a poor boy from a poor family",
      "Spare him his life from this monstrosity"
    ]
  },
  {
    id: 2,
    title: "Stairway to Heaven",
    artist: "Led Zeppelin",
    lyrics: [
      "There's a lady who's sure all that glitters is gold",
      "And she's buying a stairway to heaven",
      "When she gets there she knows",
      "If the stores are all closed",
      "With a word she can get what she came for",
      "Ooh, let me go",
      "Oh, let me go",
      "Oh, let me go"
    ]
  },
  {
    id: 3,
    title: "Hotel California",
    artist: "Eagles",
    lyrics: [
      "On a dark desert highway, cool wind in my hair",
      "Warm smell of colitas, rising up through the air",
      "Up ahead in the distance, I saw a shimmering light",
      "My head grew heavy and my sight grew dim",
      "I had to stop for the night",
      "There she stood in the doorway",
      "I heard the mission bell",
      "And I was thinking to myself"
    ]
  }
];

// Song component
const SongComponent = ({ song, onPlay, onPause, onStop, isPlaying, currentTime }) => {
  const [currentLyricIndex, setCurrentLyricIndex] = useState(0);

  useEffect(() => {
    if (isPlaying && song.lyrics.length > 0) {
      const interval = setInterval(() => {
        setCurrentLyricIndex(prev => {
          if (prev < song.lyrics.length - 1) {
            return prev + 1;
          } else {
            return prev;
          }
        });
      }, 3000); // Change lyric every 3 seconds

      return () => clearInterval(interval);
    }
  }, [isPlaying, song.lyrics.length]);

  return (
    <div className="song-container">
      <h2>{song.title}</h2>
      <p className="artist">by {song.artist}</p>

      <div className="lyrics-container">
        {song.lyrics.map((lyric, index) => (
          <p
            key={index}
            className={`lyric ${index === currentLyricIndex ? 'active' : ''}`}
          >
            {lyric}
          </p>
        ))}
      </div>

      <div className="controls">
        <button onClick={onPlay} disabled={isPlaying}>Play</button>
        <button onClick={onPause} disabled={!isPlaying}>Pause</button>
        <button onClick={onStop}>Stop</button>
      </div>
    </div>
  );
};

// Main app component
const App = () => {
  const [selectedSong, setSelectedSong] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const queryClient = new QueryClient();

  const handlePlay = () => {
    setIsPlaying(true);
  };

  const handlePause = () => {
    setIsPlaying(false);
  };

  const handleStop = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  return (
    <QueryClientProvider client={queryClient}>
      <div className="app">
        <h1>Rock Karaoke</h1>

        {!selectedSong ? (
          <div className="song-list">
            <h2>Choose a Song</h2>
            {mockSongs.map(song => (
              <div
                key={song.id}
                className="song-item"
                onClick={() => setSelectedSong(song)}
              >
                <h3>{song.title}</h3>
                <p>{song.artist}</p>
              </div>
            ))}
          </div>
        ) : (
          <SongComponent
            song={selectedSong}
            onPlay={handlePlay}
            onPause={handlePause}
            onStop={handleStop}
            isPlaying={isPlaying}
            currentTime={currentTime}
          />
        )}
      </div>
    </QueryClientProvider>
  );
};

// Render the app
const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);