import { describe, expect, test } from "bun:test";
import {
  CAMERA_PRESETS,
  LANDMARKS,
  VEHICLE_ROUTES,
  findLandmarkById,
  samplePolyline,
} from "./catalog";

describe("San Francisco landmark catalog", () => {
  test("contains all eight unique San Francisco landmarks within the island slab", () => {
    const ids = new Set(LANDMARKS.map((item) => item.id));
    expect(ids.size).toBe(8);

    for (const landmark of LANDMARKS) {
      expect(Math.abs(landmark.position[0])).toBeLessThanOrEqual(12);
      expect(Math.abs(landmark.position[2])).toBeLessThanOrEqual(12);
      expect(landmark.yearBuilt).toBeGreaterThan(1850);
      expect(landmark.yearBuilt).toBeLessThanOrEqual(2026);
      expect(landmark.zoom).toBeGreaterThan(1);
    }
  });

  test("resolves known landmarks and rejects unknown ids loudly", () => {
    const bridge = findLandmarkById("golden-gate");
    expect(bridge.name).toBe("Golden Gate Bridge");
    expect(bridge.yearBuilt).toBe(1937);

    const salesforce = findLandmarkById("salesforce-tower");
    expect(salesforce.yearBuilt).toBe(2018);
  });

  test("provides an isometric overview preset as the default entry view", () => {
    const overview = CAMERA_PRESETS[0];
    expect(overview).toBeDefined();
    expect(overview?.id).toBe("overview");
    expect(overview?.zoom).toBe(1);
  });
});

describe("Vehicle route sampling", () => {
  test("interpolates linearly along straight and multi-segment road paths", () => {
    const start = samplePolyline(
      [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      0,
    );
    expect(start.x).toBeCloseTo(0, 5);
    expect(start.z).toBeCloseTo(0, 5);

    const corner = samplePolyline(
      [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      0.5,
    );
    expect(corner.x).toBeCloseTo(10, 5);
    expect(corner.z).toBeCloseTo(0, 5);

    const threeQuarters = samplePolyline(
      [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      0.75,
    );
    expect(threeQuarters.x).toBeCloseTo(10, 5);
    expect(threeQuarters.z).toBeCloseTo(5, 5);
    expect(threeQuarters.angle).toBeCloseTo(0, 5);
  });

  test("wraps negative and overflow progress cleanly without NaN coordinates", () => {
    const a = samplePolyline(
      [
        [-5, 2],
        [5, 2],
      ],
      -0.25,
    );
    const b = samplePolyline(
      [
        [-5, 2],
        [5, 2],
      ],
      1.75,
    );
    expect(a.x).toBeCloseTo(b.x, 5);
    expect(a.z).toBeCloseTo(b.z, 5);
  });

  test("keeps every declared vehicle route strictly inside the diorama bounds", () => {
    for (const route of VEHICLE_ROUTES) {
      expect(route.points.length).toBeGreaterThanOrEqual(2);
      expect(route.speed).toBeGreaterThan(0);
      for (const [x, z] of route.points) {
        expect(Math.abs(x)).toBeLessThanOrEqual(11.5);
        expect(Math.abs(z)).toBeLessThanOrEqual(11.5);
      }
    }
  });
});
