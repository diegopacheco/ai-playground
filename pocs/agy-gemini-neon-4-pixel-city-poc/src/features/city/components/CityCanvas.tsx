import { useEffect, useRef } from "react";
import { createCityScene, type CitySceneController } from "../scene";
import type { CameraPreset, LandmarkId, LightingMode } from "../types";

type CityCanvasProps = {
  readonly lighting: LightingMode;
  readonly paused: boolean;
  readonly showPins: boolean;
  readonly crispOutlines: boolean;
  readonly activePreset: CameraPreset;
  readonly presetCounter: number;
  readonly selectedLandmark: LandmarkId | null;
  readonly onSelectLandmark: (id: LandmarkId | null) => void;
};

export function CityCanvas({
  lighting,
  paused,
  showPins,
  crispOutlines,
  activePreset,
  presetCounter,
  selectedLandmark,
  onSelectLandmark,
}: CityCanvasProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const controllerRef = useRef<CitySceneController | null>(null);
  const callbackRef = useRef(onSelectLandmark);
  callbackRef.current = onSelectLandmark;

  useEffect(() => {
    const container = hostRef.current;
    if (!container) {
      return;
    }
    const controller = createCityScene(container, lighting, {
      onSelectLandmark: (id) => callbackRef.current(id),
    });
    controllerRef.current = controller;
    return () => {
      controller.dispose();
      controllerRef.current = null;
    };
  }, []);

  useEffect(() => {
    controllerRef.current?.setLighting(lighting);
  }, [lighting]);

  useEffect(() => {
    controllerRef.current?.setPaused(paused);
  }, [paused]);

  useEffect(() => {
    controllerRef.current?.setShowPins(showPins);
  }, [showPins]);

  useEffect(() => {
    controllerRef.current?.setCrispOutlines(crispOutlines);
  }, [crispOutlines]);

  useEffect(() => {
    if (presetCounter > 0) {
      controllerRef.current?.focusPreset(activePreset);
    }
  }, [activePreset, presetCounter]);

  useEffect(() => {
    if (selectedLandmark) {
      controllerRef.current?.focusLandmark(selectedLandmark);
    }
  }, [selectedLandmark]);

  return (
    <div
      ref={hostRef}
      className="city-canvas"
      aria-label="Interactive 3D San Francisco Pixel City"
    />
  );
}
