import { Difficulty, ThemeName, GameConfig } from "./types";

interface Props {
  config: GameConfig;
  onChange: (config: GameConfig) => void;
  theme: { text: string; accent: string; boardBg: string; gridLine: string };
}

const selectStyle = (theme: Props["theme"]): React.CSSProperties => ({
  padding: "8px 12px",
  borderRadius: 4,
  border: `1px solid ${theme.gridLine}`,
  backgroundColor: theme.boardBg,
  color: theme.text,
  fontSize: 14,
  width: "100%",
});

const labelStyle = (theme: Props["theme"]): React.CSSProperties => ({
  color: theme.text,
  fontSize: 13,
  fontWeight: 600,
  marginBottom: 4,
  display: "block",
});

export default function ConfigMenu({ config, onChange, theme }: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, padding: 16, background: theme.boardBg, borderRadius: 8, border: `1px solid ${theme.gridLine}` }}>
      <div style={{ color: theme.accent, fontSize: 18, fontWeight: 700 }}>Settings</div>

      <div>
        <label style={labelStyle(theme)}>Difficulty</label>
        <select
          value={config.difficulty}
          onChange={(e) => onChange({ ...config, difficulty: e.target.value as Difficulty })}
          style={selectStyle(theme)}
        >
          <option value="easy">Easy</option>
          <option value="medium">Medium</option>
          <option value="hard">Hard</option>
        </select>
      </div>

      <div>
        <label style={labelStyle(theme)}>Theme</label>
        <select
          value={config.theme}
          onChange={(e) => onChange({ ...config, theme: e.target.value as ThemeName })}
          style={selectStyle(theme)}
        >
          <option value="classic">Classic</option>
          <option value="neon">Neon</option>
          <option value="pastel">Pastel</option>
          <option value="dark">Dark</option>
          <option value="ocean">Ocean</option>
        </select>
      </div>

      <div>
        <label style={{ ...labelStyle(theme), display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="checkbox"
            checked={config.timer_enabled}
            onChange={(e) => onChange({ ...config, timer_enabled: e.target.checked })}
          />
          Timer
        </label>
      </div>

      {config.timer_enabled && (
        <div>
          <label style={labelStyle(theme)}>Minutes</label>
          <input
            type="number"
            min={1}
            max={30}
            value={config.timer_minutes}
            onChange={(e) => onChange({ ...config, timer_minutes: parseInt(e.target.value) || 5 })}
            style={{ ...selectStyle(theme), width: 80 }}
          />
        </div>
      )}
    </div>
  );
}
