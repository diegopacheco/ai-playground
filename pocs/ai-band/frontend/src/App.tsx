import React, { useState, useRef } from 'react';
import abcjs from 'abcjs';
import './App.css';

interface MusicianOutput {
  musician: string;
  round: number;
  abc_notation: string;
}

interface CompositionResult {
  rounds: MusicianOutput[][];
  final_song: string;
}

const MUSICIAN_COLORS: Record<string, string> = {
  drums: '#e74c3c',
  bass: '#3498db',
  melody: '#2ecc71',
  lyrics: '#f39c12',
};

const MUSICIAN_ICONS: Record<string, string> = {
  drums: 'Drums',
  bass: 'Bass',
  melody: 'Melody',
  lyrics: 'Lyrics',
};

function App() {
  const [genre, setGenre] = useState('jazz funk');
  const [rounds, setRounds] = useState(2);
  const [composing, setComposing] = useState(false);
  const [result, setResult] = useState<CompositionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeMusician, setActiveMusician] = useState<string | null>(null);
  const notationRef = useRef<HTMLDivElement>(null);
  const synthRef = useRef<any>(null);

  const compose = async () => {
    setComposing(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch('http://localhost:8080/compose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ genre, rounds }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Composition failed');
      }
      const data: CompositionResult = await res.json();
      setResult(data);
      renderAbc(data.final_song);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setComposing(false);
    }
  };

  const renderAbc = (abc: string) => {
    if (notationRef.current) {
      abcjs.renderAbc(notationRef.current, abc, {
        responsive: 'resize',
        staffwidth: 700,
      });
    }
  };

  const playSong = () => {
    if (!result || !notationRef.current) return;
    if (synthRef.current) {
      synthRef.current.stop();
    }
    const visualObj = abcjs.renderAbc(notationRef.current, result.final_song, {
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
  };

  const renderMusicianAbc = (abc: string, musician: string, round: number) => {
    const id = `abc-${musician}-${round}`;
    setTimeout(() => {
      const el = document.getElementById(id);
      if (el) {
        abcjs.renderAbc(el, abc, { responsive: 'resize', staffwidth: 400 });
      }
    }, 100);
    return id;
  };

  return (
    <div className="app">
      <header className="header">
        <h1>AI Band Composer</h1>
        <p className="subtitle">Four AI musicians collaborate to compose a song</p>
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

      {composing && (
        <div className="loading">
          <div className="spinner" />
          <p>Musicians are jamming... This takes a few minutes.</p>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="results">
          <div className="band-members">
            {['drums', 'bass', 'melody', 'lyrics'].map((m) => (
              <div
                key={m}
                className={`member-card ${activeMusician === m ? 'active' : ''}`}
                style={{ borderColor: MUSICIAN_COLORS[m] }}
                onClick={() => setActiveMusician(activeMusician === m ? null : m)}
              >
                <div className="member-name" style={{ color: MUSICIAN_COLORS[m] }}>
                  {MUSICIAN_ICONS[m]}
                </div>
              </div>
            ))}
          </div>

          <div className="rounds-section">
            {result.rounds.map((round, ri) => (
              <div key={ri} className="round">
                <h3>Round {ri + 1}</h3>
                <div className="round-outputs">
                  {round
                    .filter((o) => !activeMusician || o.musician === activeMusician)
                    .map((output, oi) => (
                      <div
                        key={oi}
                        className="musician-output"
                        style={{ borderLeftColor: MUSICIAN_COLORS[output.musician] }}
                      >
                        <div className="output-header">
                          <span
                            className="musician-tag"
                            style={{ backgroundColor: MUSICIAN_COLORS[output.musician] }}
                          >
                            {output.musician}
                          </span>
                        </div>
                        <pre className="abc-text">{output.abc_notation}</pre>
                        <div id={renderMusicianAbc(output.abc_notation, output.musician, output.round)} className="abc-render" />
                      </div>
                    ))}
                </div>
              </div>
            ))}
          </div>

          <div className="final-song">
            <h2>Final Composition</h2>
            <div className="playback-controls">
              <button className="play-btn" onClick={playSong}>Play</button>
              <button className="stop-btn" onClick={stopSong}>Stop</button>
            </div>
            <div ref={notationRef} className="notation-display" />
            <pre className="abc-text final-abc">{result.final_song}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
