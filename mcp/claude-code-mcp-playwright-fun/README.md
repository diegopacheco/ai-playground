# Claude Code and MCP Playwright

## Install 

```bash
claude mcp add playwright npx @playwright/mcp@latest
```

## Using Playwright MCP

Prompt
```
using playright can you test all feature of @features.md and produce a report?
```

## Generating Tests

```
Write a playwright test script to test all features listed on @features.md the script must be called: test-ui.sh
```

Result: [test-ui.sh](test-ui.sh)

```
❯ ./test-ui.sh
Running UI tests...

Running 10 tests using 1 worker

  ✓   1 tests/ui.spec.ts:8:7 › Product Manager UI › page title is Product Manager (137ms)
  ✓   2 tests/ui.spec.ts:13:7 › Product Manager UI › Add New Product button shows and hides form (183ms)
  ✓   3 tests/ui.spec.ts:24:7 › Product Manager UI › product form has all fields (163ms)
  ✓   4 tests/ui.spec.ts:32:7 › Product Manager UI › Save Product button submits new product (296ms)
  ✓   5 tests/ui.spec.ts:45:7 › Product Manager UI › Cancel button hides form without saving (186ms)
  ✓   6 tests/ui.spec.ts:53:7 › Product Manager UI › products table has correct columns (122ms)
  ✓   7 tests/ui.spec.ts:60:7 › Product Manager UI › pre-loaded products are displayed (123ms)
  ✓   8 tests/ui.spec.ts:66:7 › Product Manager UI › View button exists on each row (123ms)
  ✓   9 tests/ui.spec.ts:73:7 › Product Manager UI › View button opens product URL in new tab (748ms)
  ✓  10 tests/ui.spec.ts:81:7 › Product Manager UI › clicking table row opens product URL in new tab (643ms)

  10 passed (3.6s)

Test complete. HTML report available at: playwright-report/index.html
```