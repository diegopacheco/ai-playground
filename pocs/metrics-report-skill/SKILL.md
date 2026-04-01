---
name: metrics-report
description: Scan an entire codebase, discover and run all test types, compute hybrid coverage, evaluate quality, and generate a full metrics report website with trends and charts.
allowed-tools: [Glob, Grep, Read, Write, Edit, Bash, Agent]
---

# Metrics Report Skill

You are a metrics analysis agent. When invoked, scan the codebase, discover tests, run them, compute coverage, evaluate quality, and produce a JSON report for the metrics React application.

IMPORTANT: Optimize for speed. Use Agent tool to parallelize independent work. Use Grep/Glob instead of reading files when possible. Batch shell commands.

## Phase 1: Setup and Detection (do all at once)

Run these in a SINGLE Bash call chained with &&:

```bash
git remote get-url origin 2>/dev/null || echo ""; git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main"; git rev-parse --short HEAD 2>/dev/null || echo "unknown"
```

Check if `metrics-report/metrics-config.json` exists. If not, create it:

```json
{
  "port": 3737,
  "testTypes": {
    "unit": true,
    "integration": true,
    "contract": true,
    "e2e": true,
    "css": false,
    "stress": false,
    "chaos": false,
    "mutation": false,
    "observability": true
  },
  "githubRemote": "origin"
}
```

Detect stacks using Glob (one call, check results):
- `pom.xml` or `build.gradle` or `build.gradle.kts` = Java/Spring Boot
- `package.json` with react = React/Node
- `manage.py` or django in `pyproject.toml` = Django/Python
- `Cargo.toml` with actix/axum/tokio = Rust

## Phase 2: Parallel Scan (use 3 Agents simultaneously)

Launch these 3 agents in PARALLEL using the Agent tool:

### Agent 1: File Count and Test Discovery

Use Glob to count all source files (skip node_modules, target, dist, build, .git, __pycache__, venv, .venv, metrics-report). Split by frontend/backend using file extension and path.

Then use Grep to find all test files in ONE pass per pattern:

```
Grep pattern="@Test|@ParameterizedTest" for Java
Grep pattern="def test_|class Test" for Python  
Grep pattern="#\[test\]|#\[cfg\(test\)\]" for Rust
Grep pattern="describe\(|it\(|test\(" for JS/TS
```

Classify each test file by type using these Grep checks (batch them):
- Integration: Grep for `@SpringBootTest|@Testcontainers|@DataJpaTest|IntegrationTest|supertest` across test files
- E2E: Grep for `@playwright/test|playwright` across test files
- Contract: Grep for `Pact|Contract|pact` across test files
- Stress: Grep for `k6/http|k6/ws` or files matching `*.k6.js`
- Chaos: Grep for `chaos-monkey|litmus|toxiproxy` across test files
- Observability: Grep for `MeterRegistry|prometheus|opentelemetry|actuator/health|tracing` in test files
- CSS: Grep for `jest-image-snapshot|percy|chromatic|toMatchImageSnapshot` in test files
- Everything else in test dirs = unit

For each test file, extract test method names and line numbers using Grep (not Read):
- Java: `Grep pattern="@Test" -A 1` to get method names
- Python: `Grep pattern="def test_"`
- Rust: `Grep pattern="fn test_|#\[test\]" -A 1`
- JS/TS: `Grep pattern="(it|test)\('" `

Return: file counts, test file list with types and test names.

### Agent 2: Git Attribution (batch)

Run a SINGLE bash command to get all test file authors at once:

```bash
for f in $(find . -path '*/test*' -name '*.java' -o -name '*.py' -o -name '*.rs' -o -name '*.ts' -o -name '*.tsx' -o -name '*.js' | grep -v node_modules | grep -v target | grep -v dist); do echo "$f:$(git log --format='%an' --diff-filter=A -- "$f" 2>/dev/null | head -1)"; done
```

Return: map of file path to author.

### Agent 3: Run Tests and Collect Coverage

Only run test types enabled in metrics-config.json. Run tests with coverage when possible. Combine test run + coverage in one command:

Java (Maven): `./mvnw test jacoco:report -q 2>&1 | tail -20`
Java (Gradle): `./gradlew test jacocoTestReport -q 2>&1 | tail -20`
Python: `pytest --tb=short -v --cov=. --cov-report=json 2>&1 | tail -30`
Rust: `cargo test 2>&1 | tail -20` then `cargo tarpaulin --out json 2>&1` if available
Node: `npm test -- --watchAll=false --coverage 2>&1 | tail -30`
E2E: `npx playwright test 2>&1 | tail -20`

Parse test results from output. Parse coverage from report files:
- Java: `target/site/jacoco/jacoco.csv`
- Python: `coverage.json`
- Rust: `tarpaulin-report.json`
- Node: `coverage/coverage-summary.json`

Return: test results (pass/fail per test), coverage data, durations.

## Phase 3: LLM Coverage Mapping (fast)

Do NOT read every file. Instead, for each test file use Grep to extract import statements in one batch:

```
Grep pattern="^import " path="test-file.java"
```

Map imports to source files. For E2E tests, Grep for route URLs (`page.goto`, `http.get`) and map to controllers. Keep it to direct imports only — do not trace full call chains.

## Phase 4: Quality Evaluation (fast)

For each test type that has tests, read ONLY 3 representative test files (not 10). Evaluate quickly:
- Assertion quality (meaningful vs trivial)
- Edge case coverage
- Test naming quality

Rate: poor/fair/good/excellent with one sentence justification.

## Phase 5: Compute Score and Generate JSON

Score (0-10):
| Criteria | Max | Calculation |
|----------|-----|-------------|
| Coverage breadth | 3 | (covered files / total source files) * 3 |
| Type diversity | 2 | 9 types=2.0, 7-8=1.7, 5-6=1.5, 3-4=1.0, 1-2=0.5 |
| Pass rate | 2 | (passing / total) * 2 |
| Test quality | 2 | Average ratings: poor=0, fair=0.7, good=1.5, excellent=2.0 |
| Code-to-test ratio | 1 | test LOC / source LOC: >0.8=1.0, >0.5=0.7, >0.3=0.4, <0.3=0.1 |

Generate `metrics-report/data/metrics-latest.json` with the full schema:

```json
{
  "timestamp": "ISO-8601",
  "repository": { "name": "", "url": "", "branch": "", "commit": "" },
  "stacks": [],
  "files": { "total": 0, "backend": 0, "frontend": 0, "testFiles": 0, "byExtension": {} },
  "tests": {
    "total": 0, "passing": 0, "failing": 0,
    "byType": {
      "unit": { "total": 0, "passing": 0, "failing": 0, "duration": 0, "files": [] },
      "integration": { "total": 0, "passing": 0, "failing": 0, "duration": 0, "files": [] },
      "contract": { "total": 0, "passing": 0, "failing": 0, "duration": 0, "files": [] },
      "e2e": { "total": 0, "passing": 0, "failing": 0, "duration": 0, "files": [] },
      "css": { "total": 0, "passing": 0, "failing": 0, "duration": 0, "files": [] },
      "stress": { "total": 0, "passing": 0, "failing": 0, "duration": 0, "files": [] },
      "chaos": { "total": 0, "passing": 0, "failing": 0, "duration": 0, "files": [] },
      "mutation": { "total": 0, "passing": 0, "failing": 0, "duration": 0, "files": [] },
      "observability": { "total": 0, "passing": 0, "failing": 0, "duration": 0, "files": [] }
    }
  },
  "failures": { "byType": {}, "details": [] },
  "coverage": { "backend": [], "frontend": [] },
  "authors": {},
  "quality": { "byType": {} },
  "score": { "total": 0, "breakdown": { "coverageBreadth": 0, "typeDiversity": 0, "passRate": 0, "testQuality": 0, "codeToTestRatio": 0 } }
}
```

Test file shape: `{ "path": "", "type": "", "testCount": 0, "passing": 0, "failing": 0, "author": "", "githubUrl": "", "tests": [{ "name": "", "status": "pass|fail", "duration": 0, "error": "", "line": 0, "githubUrl": "" }] }`

Coverage file shape: `{ "file": "", "layer": "backend|frontend", "githubUrl": "", "coverage": { "unit": { "tool": null, "llm": false } } }`

GitHub URL pattern: `https://github.com/{owner}/{repo}/blob/{branch}/{filepath}#L{line}`

## Phase 6: History and Finalize

Run in ONE Bash call:

```bash
mkdir -p metrics-report/data/history
cp metrics-report/data/metrics-latest.json "metrics-report/data/history/metrics-$(date -u +%Y-%m-%dT%H-%M-%S).json"
ls metrics-report/data/history/*.json | xargs -I{} basename {} | sort -r | head -50
```

Write the `history-index.json` with the file list from above.

If `metrics-report/metrics-application/` exists, copy data:
```bash
mkdir -p metrics-report/metrics-application/public/data/history
cp metrics-report/data/metrics-latest.json metrics-report/metrics-application/public/data/
cp metrics-report/data/history-index.json metrics-report/metrics-application/public/data/
cp metrics-report/data/history/*.json metrics-report/metrics-application/public/data/history/
```

Print: "Report generated. Run `metrics-report/run.sh` to view at http://localhost:3737"

## Rules

- Do NOT fabricate results. Use real test output and real file analysis.
- Do NOT invent files that do not exist.
- If a test type has zero tests, report zeros.
- Preserve previous history snapshots.
- JSON must be valid.
- MAXIMIZE parallelism. Use Agent tool for independent work.
- MINIMIZE file reads. Use Grep/Glob over Read when possible.
- BATCH shell commands. One Bash call with chained commands over many calls.
