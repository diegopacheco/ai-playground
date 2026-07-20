const test = require("node:test");
const assert = require("node:assert/strict");
const { containerCommand, curlCommand, duration, serviceUrl, shellQuote, statusTone } = require("../src/lib.js");

test("creates safe access commands", () => {
  assert.equal(containerCommand({ name: "api-dev" }), "podman exec -it api-dev sh");
  assert.equal(containerCommand({ name: "api dev" }), "podman exec -it 'api dev' sh");
  assert.equal(shellQuote("it's-ready"), "'it'\\''s-ready'");
});

test("creates service addresses and curl commands", () => {
  assert.equal(serviceUrl({ port: 8080 }), "http://localhost:8080");
  assert.equal(curlCommand("http://localhost:8080/health"), "curl -i 'http://localhost:8080/health'");
});

test("formats request telemetry", () => {
  assert.equal(duration(12.7), "13 ms");
  assert.equal(statusTone(204), "good");
  assert.equal(statusTone(302), "warn");
  assert.equal(statusTone(500), "danger");
});

test("manifest exposes a direct toolbar entry", () => {
  const manifest = require("../manifest.json");
  assert.equal(manifest.action.default_title, "Open Localhost Radar");
  assert.equal(manifest.action.default_popup, "src/launcher.html");
  assert.equal(manifest.devtools_page, "src/devtools.html");
  assert.equal(manifest.description, "Inspect Podman containers, localhost services, and DevTools traffic by Diego Pacheco (diegopacheco.github.io).");
});
