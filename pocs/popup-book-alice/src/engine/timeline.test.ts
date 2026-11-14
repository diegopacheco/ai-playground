import { describe, expect, test } from "bun:test";
import { accordionZ, fitDistance, layerFade, sceneZ, SET_DEPTH, stationZ, TRAVEL_TIME, travelFrame } from "./timeline";

describe("travelling through the tunnel", () => {
  test("every scene sits deeper in the same tunnel, so turning a page moves the camera forward", () => {
    for (let i = 1; i < 10; i++) expect(sceneZ(i)).toBeLessThan(sceneZ(i - 1));
    expect(stationZ(0)).toBeGreaterThan(sceneZ(0));
  });

  test("the camera starts at the old scene and ends exactly at the new one", () => {
    expect(travelFrame(0).progress).toBe(0);
    expect(travelFrame(TRAVEL_TIME).progress).toBe(1);
    expect(travelFrame(TRAVEL_TIME * 3).arrived).toBe(true);
    expect(travelFrame(TRAVEL_TIME * 0.5).arrived).toBe(false);
  });

  test("the next tunnel only opens after the camera is on its way", () => {
    expect(travelFrame(0.2).nextOpenness).toBe(0);
    expect(travelFrame(TRAVEL_TIME * 2).nextOpenness).toBe(1);
  });

  test("a closed accordion stacks every layer flat against the back, an open one puts each layer at its depth", () => {
    expect(accordionZ(0, 0)).toBe(-SET_DEPTH);
    expect(accordionZ(-3, 0)).toBe(-SET_DEPTH);
    expect(accordionZ(-3, 1)).toBeCloseTo(-3);
    expect(accordionZ(0, 1)).toBeCloseTo(0);
  });

  test("a paper layer fades away just before the camera pushes through it, and stays solid when far", () => {
    expect(layerFade(0)).toBe(0);
    expect(layerFade(-1)).toBe(0);
    expect(layerFade(5)).toBe(1);
    expect(layerFade(0.8)).toBeGreaterThan(0);
    expect(layerFade(0.8)).toBeLessThan(1);
  });

  test("a phone in portrait gets the camera further back so the book still fits", () => {
    expect(fitDistance(0.46, 3, 4.6, 40)).toBeGreaterThan(fitDistance(1.7, 5.7, 3.9, 40));
  });
});
