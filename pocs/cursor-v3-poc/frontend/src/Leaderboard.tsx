import { useEffect, useState } from "react";
import { ScoreEntry, Theme } from "./types";
import { fetchScores } from "./api";

interface Props {
  theme: Theme;
  refreshKey: number;
}

export default function Leaderboard({ theme, refreshKey }: Props) {
  const [scores, setScores] = useState<ScoreEntry[]>([]);

  useEffect(() => {
    fetchScores().then(setScores).catch(() => {});
  }, [refreshKey]);

  return (
    <div style={{ background: theme.boardBg, borderRadius: 8, border: `1px solid ${theme.gridLine}`, padding: 16, minWidth: 280 }}>
      <div style={{ color: theme.accent, fontSize: 18, fontWeight: 700, marginBottom: 12 }}>Leaderboard</div>
      {scores.length === 0 ? (
        <div style={{ color: theme.text, opacity: 0.6, fontSize: 14 }}>No scores yet. Play a game!</div>
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: `1px solid ${theme.gridLine}` }}>
              <th style={{ color: theme.text, textAlign: "left", padding: "4px 8px", fontSize: 12, opacity: 0.7 }}>#</th>
              <th style={{ color: theme.text, textAlign: "left", padding: "4px 8px", fontSize: 12, opacity: 0.7 }}>Player</th>
              <th style={{ color: theme.text, textAlign: "right", padding: "4px 8px", fontSize: 12, opacity: 0.7 }}>Score</th>
              <th style={{ color: theme.text, textAlign: "right", padding: "4px 8px", fontSize: 12, opacity: 0.7 }}>Lvl</th>
              <th style={{ color: theme.text, textAlign: "right", padding: "4px 8px", fontSize: 12, opacity: 0.7 }}>Lines</th>
            </tr>
          </thead>
          <tbody>
            {scores.slice(0, 10).map((s, i) => (
              <tr key={s.id} style={{ borderBottom: `1px solid ${theme.gridLine}` }}>
                <td style={{ color: i === 0 ? theme.accent : theme.text, padding: "6px 8px", fontSize: 14, fontWeight: i === 0 ? 700 : 400 }}>{i + 1}</td>
                <td style={{ color: i === 0 ? theme.accent : theme.text, padding: "6px 8px", fontSize: 14, fontWeight: i === 0 ? 700 : 400 }}>{s.player_name}</td>
                <td style={{ color: i === 0 ? theme.accent : theme.text, padding: "6px 8px", fontSize: 14, textAlign: "right", fontWeight: i === 0 ? 700 : 400 }}>{s.score.toLocaleString()}</td>
                <td style={{ color: theme.text, padding: "6px 8px", fontSize: 14, textAlign: "right" }}>{s.level}</td>
                <td style={{ color: theme.text, padding: "6px 8px", fontSize: 14, textAlign: "right" }}>{s.lines_cleared}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
