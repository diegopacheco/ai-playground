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
    "viewer.html",
    "viewer.css",
    "viewer.js",
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
  assert.doesNotMatch(fs.readFileSync(path.join(root, "src", "viewer.js"), "utf8"), /setAttribute\("sandbox"/)
  assert.doesNotMatch(fs.readFileSync(path.join(root, "src", "content.js"), "utf8"), /setAttribute\("sandbox"/)
})

test("raw GitHub navigation has a full-page reader", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "manifest.json"), "utf8"))
  const background = fs.readFileSync(path.join(root, "src", "background.js"), "utf8")
  assert.equal(manifest.permissions.includes("webNavigation"), true)
  assert.match(background, /raw\.githubusercontent\.com/)
  assert.match(background, /viewer\.html\?url=/)
})

test("author and logo are published", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "manifest.json"), "utf8"))
  const readme = fs.readFileSync(path.join(root, "README.md"), "utf8")
  assert.equal(manifest.author, "Diego Pacheco")
  assert.equal(manifest.homepage_url, "https://diegopacheco.github.io")
  assert.equal(manifest.description, "Renders HTML and Markdown files while browsing GitHub by Diego Pacheco(diegopacheco.github.io).")
  assert.match(readme, /assets\/github-render-logo\.svg/)
  assert.match(readme, /https:\/\/diegopacheco\.github\.io/)
})
