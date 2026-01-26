import React, { useState } from "react";
import Song from "./Song";

const songs = [
  {
    title: "Bohemian Rhapsody",
    artist: "Queen",
    audioUrl: "/audio/bohemian.mp3",
    lyrics: [
      "Is this the real life?",
      "Is this just fantasy?",
      "Caught in a landslide, no escape from reality",
      "Open your eyes, look up to the skies and see"
    ]
  },
  {
    title: "Sweet Child O' Mine",
    artist: "Guns N' Roses",
    audioUrl: "/audio/sweet_child.mp3",
    lyrics: [
      "She's got a smile that it seems to me",
      "Like a day of sun",
      "When it comes and I see",
      "The thing that I want to be"
    ]
  }
];

const App = () => {
  const [currentSong, setCurrentSong] = useState(songs[0]);

  return (
    <div style={{ padding: "20px", fontFamily: "sans-serif" }}>
      <h1>Karaoke</h1>
      <div>
        {songs.map((s, i) => (
          <button key={i} onClick={() => setCurrentSong(s)}>
            {s.title}
          </button>
        ))}
      </div>
      <Song song={currentSong} />
    </div>
  );
};

export default App;