import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { readCompanies, LOGO_DIR } from "../tools/dataset.mjs";
import { CATEGORIES } from "../web/search.js";

const companies = readCompanies();
const NO_PUBLIC_LOGO = ["ssi"];
const CORRIDOR = { south: 37.15, north: 37.85, west: -122.55, east: -121.7 };

test("every company sits in the San Francisco to San Jose corridor, otherwise the pin lands off the valley map", () => {
  for (const c of companies) {
    assert.ok(c.lat > CORRIDOR.south && c.lat < CORRIDOR.north, `${c.name} lat ${c.lat}`);
    assert.ok(c.lon > CORRIDOR.west && c.lon < CORRIDOR.east, `${c.name} lon ${c.lon}`);
  }
});

test("ids are unique kebab-case because they key logos, markers and the API", () => {
  const ids = companies.map(c => c.id);
  assert.equal(new Set(ids).size, ids.length);
  for (const id of ids) assert.match(id, /^[a-z0-9]+(-[a-z0-9]+)*$/);
});

test("every company has an address to show on click and a known category for its color", () => {
  for (const c of companies) {
    assert.ok(c.name && c.address && c.city && c.domain, c.id);
    assert.ok(CATEGORIES[c.category], `${c.id} category ${c.category}`);
  }
});

test("each category is populated so every tab has something to show", () => {
  for (const key of Object.keys(CATEGORIES)) {
    assert.ok(companies.filter(c => c.category === key).length >= 5, key);
  }
});

test("every company has a logo on disk so the map plots logos, only companies without any public logo fall back to initials", () => {
  const missing = companies.filter(c => !fs.existsSync(path.join(LOGO_DIR, `${c.id}.png`))).map(c => c.id);
  assert.deepEqual(missing, NO_PUBLIC_LOGO);
});
