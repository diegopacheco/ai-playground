import test from "node:test";
import assert from "node:assert/strict";
import { searchCompanies, normalize } from "../web/search.js";

const companies = [
  { id: "openai", name: "OpenAI", category: "ailab", address: "1455 3rd Street", city: "San Francisco", domain: "openai.com" },
  { id: "open-door", name: "Open Door", category: "tech", address: "1 Main Street", city: "San Jose", domain: "opendoor.com" },
  { id: "google", name: "Google", category: "bigtech", address: "1600 Amphitheatre Parkway", city: "Mountain View", domain: "google.com" },
  { id: "deepmind", name: "Google DeepMind", category: "ailab", address: "2000 Shoreline", city: "Mountain View", domain: "deepmind.google" },
  { id: "wb", name: "Weights & Biases", category: "aistartup", address: "400 Alabama Street", city: "San Francisco", domain: "wandb.ai" }
];
const names = list => list.map(c => c.name);

test("an exact name beats a prefix match so typing the full name jumps to that company", () => {
  assert.equal(searchCompanies(companies, "google")[0].name, "Google");
  assert.deepEqual(names(searchCompanies(companies, "google")), ["Google", "Google DeepMind"]);
});

test("search ignores case, punctuation and accents so users can type loosely", () => {
  assert.equal(normalize("Wéights & Biases!"), "weights biases");
  assert.equal(searchCompanies(companies, "OPENAI")[0].id, "openai");
  assert.equal(searchCompanies(companies, "weights biases")[0].id, "wb");
});

test("typing a city or street finds the companies there", () => {
  assert.deepEqual(names(searchCompanies(companies, "mountain view")), ["Google", "Google DeepMind"]);
  assert.deepEqual(names(searchCompanies(companies, "alabama")), ["Weights & Biases"]);
});

test("the category tab restricts results to that category", () => {
  assert.deepEqual(names(searchCompanies(companies, "google", "ailab")), ["Google DeepMind"]);
  assert.deepEqual(names(searchCompanies(companies, "", "ailab")), ["Google DeepMind", "OpenAI"]);
});

test("an empty query lists everything alphabetically and garbage finds nothing", () => {
  assert.equal(searchCompanies(companies, "").length, companies.length);
  assert.equal(searchCompanies(companies, "").at(0).name, "Google");
  assert.deepEqual(searchCompanies(companies, "zzzz"), []);
});
