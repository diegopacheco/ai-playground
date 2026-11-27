import { useState } from "react";
import { config } from "./lib/config";
import { CAMERA_PRESETS } from "./features/city/catalog";
import { CityCanvas } from "./features/city/components/CityCanvas";
import { CityHud } from "./features/city/components/CityHud";
import type {
  CameraPreset,
  LandmarkId,
  LightingMode,
} from "./features/city/types";

export function App() {
  const initialPreset = CAMERA_PRESETS[0];
  if (!initialPreset) {
    throw new Error("At least one camera preset is required");
  }

  const [lighting, setLighting] = useState<LightingMode>(config.defaultLighting);
  const [paused, setPaused] = useState(false);
  const [showPins, setShowPins] = useState(true);
  const [crispOutlines, setCrispOutlines] = useState(true);
  const [activePreset, setActivePreset] = useState<CameraPreset>(initialPreset);
  const [presetCounter, setPresetCounter] = useState(0);
  const [selectedLandmark, setSelectedLandmark] = useState<LandmarkId | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  function handleSelectPreset(preset: CameraPreset): void {
    setSelectedLandmark(null);
    setActivePreset(preset);
    setPresetCounter((c) => c + 1);
  }

  function handleSelectLandmark(id: LandmarkId | null): void {
    setSelectedLandmark(id);
  }

  return (
    <main className="app-shell">
      <CityCanvas
        lighting={lighting}
        paused={paused}
        showPins={showPins}
        crispOutlines={crispOutlines}
        activePreset={activePreset}
        presetCounter={presetCounter}
        selectedLandmark={selectedLandmark}
        onSelectLandmark={handleSelectLandmark}
      />
      <CityHud
        lighting={lighting}
        paused={paused}
        showPins={showPins}
        crispOutlines={crispOutlines}
        activePresetId={activePreset.id}
        selectedLandmark={selectedLandmark}
        drawerOpen={drawerOpen}
        onLightingChange={setLighting}
        onTogglePaused={() => setPaused((p) => !p)}
        onTogglePins={() => setShowPins((p) => !p)}
        onToggleOutlines={() => setCrispOutlines((v) => !v)}
        onSelectPreset={handleSelectPreset}
        onSelectLandmark={handleSelectLandmark}
        onToggleDrawer={() => setDrawerOpen((o) => !o)}
      />
    </main>
  );
}
