const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");

const source = name => readFileSync(path.join(__dirname, "..", "src", name), "utf8");

test("shows Playwright run and report controls", () => {
  const html = source("sidepanel.html");
  assert.match(html, /id="run"[^>]*>Run Playwright</);
  assert.match(html, /id="report"[^>]*>Show Report</);
});

test("connects the controls to the local runner", () => {
  const javascript = source("sidepanel.js");
  assert.match(javascript, /fetch\(`\$\{runnerUrl\}\/run`/);
  assert.match(javascript, /chrome\.tabs\.create\(\{ url: `\$\{runnerUrl\}\/report\/` \}\)/);
});

test("renders highlighted Playwright tokens safely", () => {
  const javascript = source("sidepanel.js");
  assert.match(javascript, /FlowPrintLib\.highlightTest\(source\)/);
  assert.match(javascript, /span\.textContent = token\.value/);
});
