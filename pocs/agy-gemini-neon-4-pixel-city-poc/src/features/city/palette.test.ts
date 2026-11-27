import { describe, expect, test } from "bun:test";
import { LIGHTING_PRESETS } from "./palette";
import { LIGHTING_MODES } from "./types";

describe("City lighting presets", () => {
  test("defines distinct lighting parameters for day, golden hour, and night", () => {
    for (const mode of LIGHTING_MODES) {
      const preset = LIGHTING_PRESETS[mode];
      expect(preset.ambientIntensity).toBeGreaterThan(0);
      expect(preset.sunIntensity).toBeGreaterThan(0);
    }

    expect(LIGHTING_PRESETS.night.lampGlow).toBeGreaterThan(
      LIGHTING_PRESETS.day.lampGlow,
    );
    expect(LIGHTING_PRESETS.night.windowGlow).toBeGreaterThan(
      LIGHTING_PRESETS.day.windowGlow,
    );
    expect(LIGHTING_PRESETS.day.background).not.toBe(
      LIGHTING_PRESETS.night.background,
    );
  });
});
