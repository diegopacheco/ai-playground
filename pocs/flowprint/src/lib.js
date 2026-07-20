(function(root) {
  const quote = value => `'${String(value).replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\r/g, "\\r").replace(/\n/g, "\\n")}'`;

  const locatorCode = locator => {
    if (!locator) return "page.locator('body')";
    if (locator.kind === "testId") return `page.getByTestId(${quote(locator.value)})`;
    if (locator.kind === "label") return `page.getByLabel(${quote(locator.value)}, { exact: true })`;
    if (locator.kind === "role") return `page.getByRole(${quote(locator.role)}, { name: ${quote(locator.value)}, exact: true })`;
    return `page.locator(${quote(locator.value)})`;
  };

  const titleFrom = value => {
    const clean = String(value || "Recorded flow").replace(/[^a-zA-Z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
    return clean.slice(0, 70) || "Recorded flow";
  };

  const urlAssertionCode = value => {
    try {
      const url = new URL(value);
      return `url => url.origin === ${quote(url.origin)} && url.pathname === ${quote(url.pathname)}`;
    } catch {
      return quote(value);
    }
  };

  const generateTest = state => {
    const steps = state?.steps || [];
    const lines = [
      "import { test, expect } from '@playwright/test';",
      "",
      `test(${quote(titleFrom(state?.title))}, async ({ page }) => {`
    ];
    let navigated = false;
    for (const step of steps) {
      if (step.type === "network") continue;
      if (step.type === "navigation") {
        if (!navigated) {
          lines.push(`  await page.goto(${quote(step.url)});`);
          navigated = true;
        } else {
          lines.push(`  await expect(page).toHaveURL(${urlAssertionCode(step.url)});`);
        }
      }
      if (step.type === "click") lines.push(`  await ${locatorCode(step.locator)}.click();`);
      if (step.type === "fill") lines.push(`  await ${locatorCode(step.locator)}.fill(${step.value === "[redacted]" ? "process.env.FLOWPRINT_SECRET || ''" : quote(step.value)});`);
      if (step.type === "press") lines.push(`  await ${locatorCode(step.locator)}.press(${quote(step.value)});`);
      if (step.type === "select") lines.push(`  await ${locatorCode(step.locator)}.selectOption(${quote(step.value)});`);
      if (step.type === "submit") lines.push(`  await ${locatorCode(step.locator)}.evaluate(form => form.requestSubmit());`);
    }
    lines.push("});", "");
    return lines.join("\n");
  };

  const highlightTest = value => {
    const source = String(value);
    const expression = /('(?:\\.|[^'\\])*')|\b(import|from|async|await|const|return|new|process|env)\b|\b(test|expect|page|getByTestId|getByLabel|getByRole|locator|goto|click|fill|press|selectOption|evaluate|toHaveURL|requestSubmit)\b|(\b\d+\b)/g;
    const tokens = [];
    let cursor = 0;
    for (const match of source.matchAll(expression)) {
      if (match.index > cursor) tokens.push({ type: "plain", value: source.slice(cursor, match.index) });
      const type = match[1] ? "string" : match[2] ? "keyword" : match[3] ? "api" : "number";
      tokens.push({ type, value: match[0] });
      cursor = match.index + match[0].length;
    }
    if (cursor < source.length) tokens.push({ type: "plain", value: source.slice(cursor) });
    return tokens;
  };

  const describeStep = step => {
    if (step.type === "navigation") return { label: "Navigate", detail: step.url };
    if (step.type === "network") return { label: `${step.method} ${step.status}`, detail: step.url };
    if (step.type === "fill") return { label: "Fill", detail: step.value === "[redacted]" ? "Sensitive value hidden" : step.value };
    if (step.type === "press") return { label: "Press", detail: step.value };
    if (step.type === "select") return { label: "Select", detail: step.value };
    if (step.type === "submit") return { label: "Submit", detail: step.locator?.value || "form" };
    return { label: "Click", detail: step.text || step.locator?.value || "element" };
  };

  const api = { describeStep, generateTest, highlightTest, locatorCode, quote, titleFrom, urlAssertionCode };
  root.FlowPrintLib = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
