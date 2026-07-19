const test = require("node:test")
const assert = require("node:assert/strict")
const fs = require("node:fs")
const path = require("node:path")

const root = path.resolve(__dirname, "..")

test("manifest uses the current extension platform", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "manifest.json"), "utf8"))
  assert.equal(manifest.manifest_version, 3)
  assert.equal(manifest.permissions.includes("storage"), true)
  assert.equal(manifest.content_scripts[0].matches.includes("https://github.com/*/*/blob/*"), true)
})

test("manifest files exist", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "manifest.json"), "utf8"))
  const files = [
    manifest.background.service_worker,
    manifest.action.default_popup,
    ...manifest.sandbox.pages,
    ...manifest.web_accessible_resources.flatMap(entry => entry.resources),
    ...manifest.content_scripts.flatMap(entry => [...entry.js, ...entry.css])
  ]
  for (const file of files) assert.equal(fs.existsSync(path.join(root, "src", file)), true, file)
})

test("HTML viewer runs in an isolated sandbox", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "manifest.json"), "utf8"))
  assert.equal(manifest.sandbox.pages.includes("html-viewer.html"), true)
  assert.match(manifest.content_security_policy.sandbox, /sandbox allow-scripts/)
  assert.doesNotMatch(manifest.content_security_policy.sandbox, /allow-same-origin/)
})
