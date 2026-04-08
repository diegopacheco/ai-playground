import { useState, useCallback } from "react";
import {
  createRouter,
  createRoute,
  createRootRoute,
  RouterProvider,
  Outlet,
} from "@tanstack/react-router";
import { GameConfig, ThemeName, THEMES } from "./types";
import { useGame } from "./useGame";
import GameBoard from "./GameBoard";
import NextPiece from "./NextPiece";
import ConfigMenu from "./ConfigMenu";
import Leaderboard from "./Leaderboard";
import { submitScore } from "./api";

const rootRoute = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  return (
    <div>
      <Outlet />
    </div>
  );
}

function HomePage() {
  const [config, setConfig] = useState<GameConfig>({
    difficulty: "medium",
    theme: "classic",
    timer_enabled: false,
    timer_minutes: 5,
  });
  const [view, setView] = useState<"menu" | "game" | "config" | "leaderboard">("menu");
  const [playerName, setPlayerName] = useState("Player");
  const [refreshKey, setRefreshKey] = useState(0);
  const theme = THEMES[config.theme as ThemeName];

  const { state, startGame } = useGame(
    config.difficulty as any,
    config.theme as ThemeName,
    config.timer_enabled,
    config.timer_minutes
  );

  const handleStart = useCallback(() => {
    startGame();
    setView("game");
  }, [startGame]);

  const handleGameOver = useCallback(async () => {
    if (state.score > 0) {
      await submitScore({
        player_name: playerName,
        score: state.score,
        level: state.level,
        lines_cleared: state.linesCleared,
        difficulty: config.difficulty,
      }).catch(() => {});
      setRefreshKey((k) => k + 1);
    }
  }, [state.score, state.level, state.linesCleared, playerName, config.difficulty]);

  const btnStyle: React.CSSProperties = {
    padding: "12px 32px",
    fontSize: 16,
    fontWeight: 700,
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
    backgroundColor: theme.accent,
    color: "#fff",
    transition: "opacity 0.2s",
    minWidth: 200,
  };

  const containerStyle: React.CSSProperties = {
    minHeight: "100vh",
    backgroundColor: theme.background,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
  };

  if (view === "menu") {
    return (
      <div style={containerStyle}>
        <div style={{ marginTop: 80, display: "flex", flexDirection: "column", alignItems: "center", gap: 24 }}>
          <h1 style={{ color: theme.accent, fontSize: 48, fontWeight: 900, margin: 0, letterSpacing: 4 }}>TETRIS</h1>
          <p style={{ color: theme.text, opacity: 0.7, margin: 0 }}>A classic puzzle game</p>
          <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 8 }}>
            <label style={{ color: theme.text, fontSize: 13 }}>Player Name</label>
            <input
              value={playerName}
              onChange={(e) => setPlayerName(e.target.value)}
              style={{ padding: "8px 12px", borderRadius: 4, border: `1px solid ${theme.gridLine}`, backgroundColor: theme.boardBg, color: theme.text, fontSize: 14, width: 200 }}
            />
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 8 }}>
            <button style={btnStyle} onClick={handleStart}>Play Game</button>
            <button style={{ ...btnStyle, backgroundColor: "transparent", border: `2px solid ${theme.accent}`, color: theme.accent }} onClick={() => setView("config")}>Settings</button>
            <button style={{ ...btnStyle, backgroundColor: "transparent", border: `2px solid ${theme.accent}`, color: theme.accent }} onClick={() => { setRefreshKey((k) => k + 1); setView("leaderboard"); }}>Leaderboard</button>
          </div>
        </div>
      </div>
    );
  }

  if (view === "config") {
    return (
      <div style={containerStyle}>
        <div style={{ marginTop: 40, width: 320 }}>
          <ConfigMenu config={config} onChange={setConfig} theme={theme} />
          <button style={{ ...btnStyle, marginTop: 16, width: "100%" }} onClick={() => setView("menu")}>Back</button>
        </div>
      </div>
    );
  }

  if (view === "leaderboard") {
    return (
      <div style={containerStyle}>
        <div style={{ marginTop: 40, width: 400 }}>
          <Leaderboard theme={theme} refreshKey={refreshKey} />
          <button style={{ ...btnStyle, marginTop: 16, width: "100%" }} onClick={() => setView("menu")}>Back</button>
        </div>
      </div>
    );
  }

  if (state.gameOver && state.score > 0) {
    handleGameOver();
  }

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div style={containerStyle}>
      <div style={{ marginTop: 20, display: "flex", gap: 24, alignItems: "flex-start" }}>
        <div>
          <GameBoard board={state.board} currentPiece={state.currentPiece} currentPos={state.currentPos} theme={theme} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16, minWidth: 180 }}>
          <div style={{ background: theme.boardBg, borderRadius: 8, border: `1px solid ${theme.gridLine}`, padding: 16 }}>
            <div style={{ color: theme.text, fontSize: 13, opacity: 0.7 }}>Score</div>
            <div style={{ color: theme.accent, fontSize: 28, fontWeight: 700 }}>{state.score.toLocaleString()}</div>
          </div>
          <div style={{ background: theme.boardBg, borderRadius: 8, border: `1px solid ${theme.gridLine}`, padding: 16 }}>
            <div style={{ color: theme.text, fontSize: 13, opacity: 0.7 }}>Level</div>
            <div style={{ color: theme.accent, fontSize: 24, fontWeight: 700 }}>{state.level + 1} / 10</div>
          </div>
          <div style={{ background: theme.boardBg, borderRadius: 8, border: `1px solid ${theme.gridLine}`, padding: 16 }}>
            <div style={{ color: theme.text, fontSize: 13, opacity: 0.7 }}>Lines</div>
            <div style={{ color: theme.text, fontSize: 20, fontWeight: 600 }}>{state.linesCleared}</div>
          </div>
          {config.timer_enabled && (
            <div style={{ background: theme.boardBg, borderRadius: 8, border: `1px solid ${theme.gridLine}`, padding: 16 }}>
              <div style={{ color: theme.text, fontSize: 13, opacity: 0.7 }}>Time</div>
              <div style={{ color: state.timerSeconds < 30 ? "#f00" : theme.text, fontSize: 20, fontWeight: 600 }}>{formatTime(state.timerSeconds)}</div>
            </div>
          )}
          <div style={{ background: theme.boardBg, borderRadius: 8, border: `1px solid ${theme.gridLine}`, padding: 16 }}>
            <div style={{ color: theme.text, fontSize: 13, opacity: 0.7, marginBottom: 8 }}>Next</div>
            <NextPiece piece={state.nextPiece} theme={theme} />
          </div>
          <div style={{ background: theme.boardBg, borderRadius: 8, border: `1px solid ${theme.gridLine}`, padding: 12 }}>
            <div style={{ color: theme.text, fontSize: 11, opacity: 0.5, lineHeight: 1.6 }}>
              Arrow keys: Move<br />
              Up: Rotate<br />
              Space: Hard drop<br />
              Down: Soft drop
            </div>
          </div>
          {state.gameOver && (
            <div style={{ textAlign: "center" }}>
              <div style={{ color: theme.accent, fontSize: 20, fontWeight: 700, marginBottom: 12 }}>Game Over!</div>
              <button style={btnStyle} onClick={handleStart}>Play Again</button>
              <button style={{ ...btnStyle, marginTop: 8, backgroundColor: "transparent", border: `2px solid ${theme.accent}`, color: theme.accent }} onClick={() => setView("menu")}>Menu</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: HomePage,
});

const routeTree = rootRoute.addChildren([indexRoute]);
const router = createRouter({ routeTree });

export default function App() {
  return <RouterProvider router={router} />;
}
