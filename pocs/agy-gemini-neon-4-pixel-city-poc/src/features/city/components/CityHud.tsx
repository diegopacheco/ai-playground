import { CAMERA_PRESETS, LANDMARKS, findLandmarkById } from "../catalog";
import {
  LIGHTING_MODES,
  type CameraPreset,
  type LandmarkId,
  type LightingMode,
} from "../types";

type CityHudProps = {
  readonly lighting: LightingMode;
  readonly paused: boolean;
  readonly showPins: boolean;
  readonly crispOutlines: boolean;
  readonly activePresetId: CameraPreset["id"];
  readonly selectedLandmark: LandmarkId | null;
  readonly drawerOpen: boolean;
  readonly onLightingChange: (mode: LightingMode) => void;
  readonly onTogglePaused: () => void;
  readonly onTogglePins: () => void;
  readonly onToggleOutlines: () => void;
  readonly onSelectPreset: (preset: CameraPreset) => void;
  readonly onSelectLandmark: (id: LandmarkId | null) => void;
  readonly onToggleDrawer: () => void;
};

function lightingLabel(mode: LightingMode): string {
  switch (mode) {
    case "day":
      return "Day";
    case "golden":
      return "Golden";
    case "night":
      return "Night";
    default: {
      const unreachable: never = mode;
      return unreachable;
    }
  }
}

export function CityHud({
  lighting,
  paused,
  showPins,
  crispOutlines,
  activePresetId,
  selectedLandmark,
  drawerOpen,
  onLightingChange,
  onTogglePaused,
  onTogglePins,
  onToggleOutlines,
  onSelectPreset,
  onSelectLandmark,
  onToggleDrawer,
}: CityHudProps) {
  const landmarkInfo = selectedLandmark ? findLandmarkById(selectedLandmark) : null;
  const themeClass = lighting === "night" ? "hud-night" : "hud-light";

  function stepLandmark(delta: number): void {
    if (!selectedLandmark) {
      const first = LANDMARKS[0];
      if (first) {
        onSelectLandmark(first.id);
      }
      return;
    }
    const idx = LANDMARKS.findIndex((item) => item.id === selectedLandmark);
    const nextIdx = (idx + delta + LANDMARKS.length) % LANDMARKS.length;
    const next = LANDMARKS[nextIdx];
    if (next) {
      onSelectLandmark(next.id);
    }
  }

  return (
    <div className={`hud-layer ${themeClass}`}>
      <header className="hud-header">
        <div className="brand-badge">
          <span className="brand-dot" />
          <div>
            <h1 className="brand-title">SF PIXEL CITY</h1>
            <p className="brand-subtitle">37.7749° N, 122.4194° W</p>
          </div>
        </div>

        <div className="header-actions">
          <div className="segmented-group" role="group" aria-label="Lighting mode">
            {LIGHTING_MODES.map((mode) => (
              <button
                key={mode}
                type="button"
                className={lighting === mode ? "pill-btn active" : "pill-btn"}
                onClick={() => onLightingChange(mode)}
              >
                {lightingLabel(mode)}
              </button>
            ))}
          </div>

          <button
            type="button"
            className={drawerOpen ? "pill-btn active" : "pill-btn"}
            onClick={onToggleDrawer}
          >
            Landmarks ({LANDMARKS.length})
          </button>
        </div>
      </header>

      {drawerOpen && (
        <aside className="landmark-drawer" aria-label="San Francisco landmarks">
          <div className="drawer-header">
            <span className="drawer-kicker">CITY GUIDE</span>
            <button
              type="button"
              className="icon-btn"
              onClick={onToggleDrawer}
              aria-label="Close landmarks list"
            >
              ×
            </button>
          </div>
          <ul className="landmark-list">
            {LANDMARKS.map((item) => (
              <li key={item.id}>
                <button
                  type="button"
                  className={
                    selectedLandmark === item.id
                      ? "landmark-row active"
                      : "landmark-row"
                  }
                  onClick={() => onSelectLandmark(item.id)}
                >
                  <span className="landmark-row-name">{item.name}</span>
                  <span className="landmark-row-district">{item.district}</span>
                </button>
              </li>
            ))}
          </ul>
        </aside>
      )}

      {landmarkInfo && (
        <section className="inspector-card" aria-label="Selected landmark">
          <div className="inspector-top">
            <div>
              <span className="inspector-district">
                {landmarkInfo.district} · Est. {landmarkInfo.yearBuilt}
              </span>
              <h2 className="inspector-title">{landmarkInfo.name}</h2>
            </div>
            <button
              type="button"
              className="icon-btn"
              onClick={() => onSelectLandmark(null)}
              aria-label="Close landmark details"
            >
              ×
            </button>
          </div>
          <p className="inspector-desc">{landmarkInfo.description}</p>
          <div className="inspector-nav">
            <button
              type="button"
              className="pill-btn"
              onClick={() => stepLandmark(-1)}
            >
              Prev
            </button>
            <button
              type="button"
              className="pill-btn"
              onClick={() => stepLandmark(1)}
            >
              Next
            </button>
          </div>
        </section>
      )}

      <footer className="hud-dock">
        <div className="segmented-group" role="group" aria-label="Camera views">
          {CAMERA_PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              className={
                activePresetId === preset.id ? "pill-btn active" : "pill-btn"
              }
              onClick={() => onSelectPreset(preset)}
            >
              {preset.label}
            </button>
          ))}
        </div>

        <div className="segmented-group" role="group" aria-label="Scene controls">
          <button
            type="button"
            className={showPins ? "pill-btn active" : "pill-btn"}
            onClick={onTogglePins}
          >
            Pins
          </button>
          <button
            type="button"
            className={crispOutlines ? "pill-btn active" : "pill-btn"}
            onClick={onToggleOutlines}
          >
            Ink
          </button>
          <button
            type="button"
            className={paused ? "pill-btn active" : "pill-btn"}
            onClick={onTogglePaused}
          >
            {paused ? "Resume" : "Pause"}
          </button>
        </div>
      </footer>
    </div>
  );
}
