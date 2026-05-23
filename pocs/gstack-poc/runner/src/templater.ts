import type { Action, RunResult, Selector } from "./types.ts";

export function actionLogToScript(run: RunResult): string {
  const testName = deriveTestName(run.prompt);
  const successful = run.log.filter((e) => e.result.status === "ok");

  const lines: string[] = [];
  lines.push("import { test, expect } from '@playwright/test';");
  lines.push("");
  lines.push(`test(${quote(testName)}, async ({ page }) => {`);
  lines.push(`  await page.goto(${quote(run.url)});`);

  for (const entry of successful) {
    const statement = renderAction(entry.action);
    if (statement !== null) lines.push(`  ${statement}`);
  }

  if (run.stopReason !== "done") {
    lines.push(`  // stopped early: ${run.stopReason}`);
  }
  lines.push("});");
  lines.push("");
  return lines.join("\n");
}

function renderAction(action: Action): string | null {
  switch (action.tool) {
    case "click":
      return `await ${renderLocator(action.selector)}.click();`;
    case "type":
      return `await ${renderLocator(action.selector)}.fill(${quote(action.text)});`;
    case "wait_for":
      return `await ${renderLocator(action.selector)}.waitFor();`;
    case "assert_text":
      return `await expect(${renderLocator(action.selector)}).toContainText(${quote(action.text)});`;
    case "screenshot":
    case "done":
      return null;
  }
}

function renderLocator(selector: Selector): string {
  switch (selector.kind) {
    case "role":
      return selector.name !== undefined
        ? `page.getByRole(${quote(selector.role)}, { name: ${quote(selector.name)} })`
        : `page.getByRole(${quote(selector.role)})`;
    case "placeholder":
      return `page.getByPlaceholder(${quote(selector.text)})`;
    case "text":
      return `page.getByText(${quote(selector.text)})`;
    case "label":
      return `page.getByLabel(${quote(selector.text)})`;
    case "test_id":
      return `page.getByTestId(${quote(selector.id)})`;
  }
}

function quote(value: string): string {
  if (!/['\\\n\r\t]/.test(value)) {
    return `'${value}'`;
  }
  if (!/["\\\n\r]/.test(value)) {
    return `"${value}"`;
  }
  return JSON.stringify(value);
}

function deriveTestName(prompt: string): string {
  const collapsed = prompt.replace(/\s+/g, " ").trim();
  if (collapsed.length <= 80) return collapsed;
  return collapsed.slice(0, 77) + "...";
}
