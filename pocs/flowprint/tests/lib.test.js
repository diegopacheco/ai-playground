const test = require("node:test");
const assert = require("node:assert/strict");
const { describeStep, generateTest, locatorCode, quote, titleFrom, urlAssertionCode } = require("../src/lib.js");

test("quotes JavaScript strings safely", () => {
  assert.equal(quote("Diego's flow\nnext"), "'Diego\\'s flow\\nnext'");
});

test("builds Playwright locators", () => {
  assert.equal(locatorCode({ kind: "testId", value: "save" }), "page.getByTestId('save')");
  assert.equal(locatorCode({ kind: "label", value: "Email" }), "page.getByLabel('Email', { exact: true })");
  assert.equal(locatorCode({ kind: "role", role: "button", value: "Save" }), "page.getByRole('button', { name: 'Save', exact: true })");
  assert.equal(locatorCode({ kind: "css", value: "#email" }), "page.locator('#email')");
});

test("generates stable actions and navigation assertions", () => {
  const output = generateTest({
    title: "Checkout | Store",
    steps: [
      { type: "navigation", url: "http://localhost:3000" },
      { type: "fill", locator: { kind: "label", value: "Email" }, value: "a@b.com" },
      { type: "press", locator: { kind: "label", value: "Email" }, value: "Enter" },
      { type: "click", locator: { kind: "role", role: "button", value: "Pay" } },
      { type: "navigation", url: "http://localhost:3000/receipt?id=42&token=changing" },
      { type: "network", method: "POST", url: "http://localhost:3000/api/pay", status: 201 },
      { type: "network", method: "POST", url: "http://localhost:3000/api/pay", status: 201 }
    ]
  });
  assert.match(output, /test\('Checkout Store'/);
  assert.match(output, /page\.getByLabel\('Email', \{ exact: true \}\)\.fill\('a@b\.com'\)/);
  assert.match(output, /page\.getByLabel\('Email', \{ exact: true \}\)\.press\('Enter'\)/);
  assert.match(output, /page\.getByRole\('button', \{ name: 'Pay', exact: true \}\)\.click/);
  assert.match(output, /url => url\.origin === 'http:\/\/localhost:3000' && url\.pathname === '\/receipt'/);
  assert.doesNotMatch(output, /api\/pay/);
  assert.doesNotMatch(output, /changing/);
});

test("builds stable URL assertions", () => {
  assert.equal(urlAssertionCode("https://site.test/orders?id=42"), "url => url.origin === 'https://site.test' && url.pathname === '/orders'");
});

test("describes entries for the visual ledger", () => {
  assert.deepEqual(describeStep({ type: "network", method: "GET", status: 200, url: "/api" }), { label: "GET 200", detail: "/api" });
  assert.deepEqual(describeStep({ type: "press", value: "Enter" }), { label: "Press", detail: "Enter" });
  assert.equal(titleFrom("  A / B  "), "A B");
});

test("keeps captured passwords out of generated code", () => {
  const output = generateTest({
    title: "Sign in",
    steps: [{ type: "fill", locator: { kind: "label", value: "Password" }, value: "[redacted]" }]
  });
  assert.match(output, /process\.env\.FLOWPRINT_SECRET/);
  assert.doesNotMatch(output, /fill\('\[redacted\]'\)/);
});

test("manifest credits the author", () => {
  const manifest = require("../manifest.json");
  assert.equal(manifest.description, "Record browser flows and turn them into Playwright tests. GitHub by Diego Pacheco (diegopacheco.github.io).");
});
