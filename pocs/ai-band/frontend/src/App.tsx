import React, { useState, useRef, useEffect, useCallback } from 'react';
import abcjs from 'abcjs';
import './App.css';

interface MusicianOutput {
  musician: string;
  round: number;
  abc_notation: string;
}

const MUSICIAN_COLORS: Record<string, string> = {
  drums: '#e74c3c',
  bass: '#3498db',
  guitar: '#e67e22',
  melody: '#2ecc71',
  lyrics: '#f39c12',
  singer: '#9b59b6',
};

function App() {
  const [genre, setGenre] = useState('jazz funk');
  const [lyricsTheme, setLyricsTheme] = useState('');
  const [rounds, setRounds] = useState(2);
  const [composing, setComposing] = useState(false);
  const [outputs, setOutputs] = useState<MusicianOutput[]>([]);
  const [thinking, setThinking] = useState<string | null>(null);
  const [thinkingRound, setThinkingRound] = useState<number>(0);
  const [finalSong, setFinalSong] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>('live');
  const notationRef = useRef<HTMLDivElement>(null);
  const synthRef = useRef<any>(null);

  const compose = useCallback(() => {
    setComposing(true);
    setError(null);
    setOutputs([]);
    setFinalSong(null);
    setThinking(null);
    setActiveTab('live');

    const params = new URLSearchParams({ genre, rounds: String(rounds) });
    if (lyricsTheme.trim()) params.set('lyrics_theme', lyricsTheme.trim());
    const url = `http://localhost:8080/compose?${params.toString()}`;
    const es = new EventSource(url);

    es.addEventListener('thinking', (e: MessageEvent) => {
      const data = JSON.parse(e.data);
      setThinking(data.musician);
      setThinkingRound(data.round);
    });

    es.addEventListener('done', (e: MessageEvent) => {
      const data = JSON.parse(e.data);
      setThinking(null);
      setOutputs(prev => [...prev, {
        musician: data.musician,
        round: data.round,
        abc_notation: data.abc_notation,
      }]);
    });

    es.addEventListener('final', (e: MessageEvent) => {
      const data = JSON.parse(e.data);
      setFinalSong(data.final_song);
      setComposing(false);
      setActiveTab('song');
      es.close();
    });

    es.addEventListener('error', (e: MessageEvent) => {
      if (e.data) {
        const data = JSON.parse(e.data);
        setError(data.message);
      }
      setComposing(false);
      es.close();
    });

    es.onerror = () => {
      if (composing) {
        setError('Connection lost');
        setComposing(false);
      }
      es.close();
    };
  }, [genre, rounds, lyricsTheme]);

  useEffect(() => {
    if (finalSong && notationRef.current && activeTab === 'song') {
      abcjs.renderAbc(notationRef.current, finalSong, {
        responsive: 'resize',
        staffwidth: 700,
      });
    }
  }, [finalSong, activeTab]);

  useEffect(() => {
    outputs.forEach((o) => {
      const id = `abc-${o.musician}-${o.round}`;
      setTimeout(() => {
        const el = document.getElementById(id);
        if (el && el.children.length === 0) {
          abcjs.renderAbc(el, o.abc_notation, { responsive: 'resize', staffwidth: 500 });
        }
      }, 50);
    });
  }, [outputs, activeTab]);

  const playSong = () => {
    if (!finalSong || !notationRef.current) return;
    if (synthRef.current) {
      synthRef.current.stop();
    }
    const visualObj = abcjs.renderAbc(notationRef.current, finalSong, {
      responsive: 'resize',
      staffwidth: 700,
    });
    if (visualObj && visualObj.length > 0) {
      const synth = new abcjs.synth.CreateSynth();
      synth.init({ visualObj: visualObj[0] }).then(() => {
        synth.prime().then(() => {
          synth.start();
          synthRef.current = synth;
        });
      });
    }
  };

  const stopSong = () => {
    if (synthRef.current) {
      synthRef.current.stop();
      synthRef.current = null;
    }
    window.speechSynthesis.cancel();
  };

  const singLyrics = () => {
    const lyricsOutputs = outputs.filter(o => o.musician === 'lyrics');
    if (lyricsOutputs.length === 0) return;
    window.speechSynthesis.cancel();
    const lastLyrics = lyricsOutputs[lyricsOutputs.length - 1].abc_notation;
    const lines = lastLyrics.split('\n')
      .filter(l => l.trim().startsWith('w:'))
      .map(l => l.replace(/^w:\s*/, '').replace(/[|~\-_*]/g, ' ').trim())
      .filter(l => l.length > 0);
    const text = lines.join('. ');
    if (!text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.8;
    utterance.pitch = 1.1;
    const voices = window.speechSynthesis.getVoices();
    const english = voices.find(v => v.lang.startsWith('en'));
    if (english) utterance.voice = english;
    window.speechSynthesis.speak(utterance);
  };

  const playWithSinging = () => {
    playSong();
    setTimeout(() => singLyrics(), 500);
  };

  const groupedByRound: Record<number, MusicianOutput[]> = {};
  outputs.forEach(o => {
    if (!groupedByRound[o.round]) groupedByRound[o.round] = [];
    groupedByRound[o.round].push(o);
  });

  const musicians = ['drums', 'bass', 'guitar', 'melody', 'lyrics', 'singer'];

  return (
    <div className="app">
      <header className="header">
        <h1>AI Band Composer</h1>
        <p className="subtitle">Six AI musicians collaborate to compose a song</p>
      </header>

      <div className="controls">
        <div className="input-group">
          <label>Genre / Style</label>
          <input
            type="text"
            value={genre}
            onChange={(e) => setGenre(e.target.value)}
            placeholder="e.g. jazz funk, rock, classical"
          />
        </div>
        <div className="input-group">
          <label>Lyrics Theme / Influence</label>
          <input
            type="text"
            value={lyricsTheme}
            onChange={(e) => setLyricsTheme(e.target.value)}
            placeholder="e.g. love, space travel, rainy days"
          />
        </div>
        <div className="input-group">
          <label>Rounds</label>
          <select value={rounds} onChange={(e) => setRounds(Number(e.target.value))}>
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={3}>3</option>
          </select>
        </div>
        <button className="compose-btn" onClick={compose} disabled={composing}>
          {composing ? 'Composing...' : 'Compose'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {(composing || outputs.length > 0) && (
        <>
          <div className="status-bar">
            {musicians.map(m => {
              const done = outputs.filter(o => o.musician === m);
              const isThinking = thinking === m;
              return (
                <div key={m} className={`status-item ${isThinking ? 'pulse' : ''} ${done.length > 0 ? 'completed' : ''}`}>
                  <div className="status-dot" style={{ backgroundColor: isThinking ? MUSICIAN_COLORS[m] : done.length > 0 ? MUSICIAN_COLORS[m] : '#444' }} />
                  <span className="status-label">{m}</span>
                  {isThinking && <span className="status-state">R{thinkingRound} thinking...</span>}
                  {!isThinking && done.length > 0 && <span className="status-state">R{done[done.length-1].round} done</span>}
                </div>
              );
            })}
          </div>

          <div className="tabs">
            <button className={`tab ${activeTab === 'live' ? 'active' : ''}`} onClick={() => setActiveTab('live')}>Live Progress</button>
            {['drums', 'bass', 'guitar', 'melody', 'lyrics', 'singer'].map(m => (
              <button
                key={m}
                className={`tab ${activeTab === m ? 'active' : ''}`}
                style={activeTab === m ? { borderBottomColor: MUSICIAN_COLORS[m] } : {}}
                onClick={() => setActiveTab(m)}
                disabled={outputs.filter(o => o.musician === m).length === 0}
              >
                {m.charAt(0).toUpperCase() + m.slice(1)}
              </button>
            ))}
            <button className={`tab ${activeTab === 'song' ? 'active' : ''}`} onClick={() => setActiveTab('song')} disabled={!finalSong}>
              Final Song
            </button>
          </div>

          <div className="tab-content">
            {activeTab === 'live' && (
              <div className="live-feed">
                {Object.entries(groupedByRound).map(([round, items]) => (
                  <div key={round} className="round">
                    <h3>Round {round}</h3>
                    <div className="round-outputs">
                      {items.map((output, i) => (
                        <div key={i} className="musician-output" style={{ borderLeftColor: MUSICIAN_COLORS[output.musician] }}>
                          <span className="musician-tag" style={{ backgroundColor: MUSICIAN_COLORS[output.musician] }}>{output.musician}</span>
                          <pre className="abc-text">{output.abc_notation}</pre>
                          <div id={`abc-${output.musician}-${output.round}`} className="abc-render" />
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
                {thinking && (
                  <div className="thinking-indicator">
                    <div className="spinner" />
                    <span style={{ color: MUSICIAN_COLORS[thinking] }}>{thinking}</span> is composing (Round {thinkingRound})...
                  </div>
                )}
              </div>
            )}

            {musicians.map(m => activeTab === m && (
              <div key={m} className="musician-tab">
                {outputs.filter(o => o.musician === m).map((output, i) => (
                  <div key={i} className="musician-output" style={{ borderLeftColor: MUSICIAN_COLORS[m] }}>
                    <h4>Round {output.round}</h4>
                    <pre className="abc-text">{output.abc_notation}</pre>
                    <div id={`abc-${output.musician}-${output.round}`} className="abc-render" />
                  </div>
                ))}
              </div>
            ))}

            {activeTab === 'song' && finalSong && (
              <div className="final-song">
                <div className="playback-controls">
                  <button className="play-btn" onClick={playSong}>Play Music</button>
                  <button className="sing-btn" onClick={playWithSinging}>Play + Sing</button>
                  <button className="stop-btn" onClick={stopSong}>Stop</button>
                </div>
                <div ref={notationRef} className="notation-display" />
                <pre className="abc-text final-abc">{finalSong}</pre>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default App;
