import test from "node:test";
import assert from "node:assert/strict";
import { distanceKm } from "../tools/dataset.mjs";

test("distance guards geocoder results: San Francisco to San Jose is about 68 km", () => {
  const km = distanceKm({ lat: 37.7749, lon: -122.4194 }, { lat: 37.3382, lon: -121.8863 });
  assert.ok(km > 65 && km < 72, `${km}`);
});

test("the same point is zero km apart", () => {
  assert.equal(distanceKm({ lat: 37.4, lon: -122 }, { lat: 37.4, lon: -122 }), 0);
});
