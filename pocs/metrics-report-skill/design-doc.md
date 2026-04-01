# Metrics Report Skill - Design Document

## Overview

A Claude Code skill that reads an entire codebase, identifies all test types, runs them, collects metrics, computes coverage (tool-based + LLM-based), and produces a full React/Node/TypeScript metrics application with historical trends, charts, search, and GitHub integration.

## Supported Stacks

| Stack | Versions | Build Tools |
|-------|----------|-------------|
| Java + Spring Boot | Java 8-25, Spring Boot 3.x/4.x | Maven (`./mvnw`), Gradle (`./gradlew`) |
| React + Node | Node/Bun | npm, bun |
| Django + Python | Django 6, Python 3.13/3.14 | pytest, manage.py |
| Rust | 1.93+, edition 2024+, tokio, actix/axum | cargo |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   SKILL EXECUTION                    │
│                                                      │
│  Phase 1: Setup + Stack Detection                    │
│  Phase 2: Parallel Scan (3 agents)                   │
│    ├─ Agent 1: File Count + Test Discovery           │
│    ├─ Agent 2: Git Attribution                       │
│    └─ Agent 3: Test Execution (SEQUENTIAL)           │
│         └─ Backend → Frontend → Other, 1 type/time   │
│  Phase 3: LLM Coverage Mapping                       │
│  Phase 4: Quality Evaluation                         │
│  Phase 5: Score Computation + JSON Output            │
│  Phase 6: History Snapshot + Auto-run Website         │
│                                                      │
│  Progress: [Phase X/6] and [Test X/Y] throughout     │
└──────────────────┬──────────────────────────────────┘
                   │ writes JSON
                   ▼
┌─────────────────────────────────────────────────────┐
│           metrics-report/                            │
│                                                      │
│  data/                                               │
│    ├── metrics-YYYY-MM-DDTHH-mm-ss.json             │
│    ├── metrics-latest.json                           │
│    └── history/                                      │
│         └── <timestamped snapshots>                  │
│                                                      │
│  metrics-application/                                │
│    ├── package.json                                  │
│    ├── tsconfig.json                                 │
│    ├── src/                                          │
│    │    ├── App.tsx                                  │
│    │    ├── components/                              │
│    │    ├── pages/                                   │
│    │    └── types/                                   │
│    └── public/                                       │
│                                                      │
│  metrics-config.json                                 │
│  run.sh                                              │
│  stop.sh                                             │
└─────────────────────────────────────────────────────┘
```

## Step-by-Step Flow

### Step 1: Stack Detection

Auto-detect by presence of:
- `pom.xml` / `build.gradle` / `build.gradle.kts` → Java/Spring Boot
- `package.json` with react dependency → React/Node
- `manage.py` / `settings.py` / `pyproject.toml` with django → Django/Python
- `Cargo.toml` with actix/axum/tokio → Rust

Multiple stacks can coexist (e.g., Java backend + React frontend). Each is scanned independently and results are merged.

### Step 2: Codebase Scan

Read every file in the repository. Collect:
- Total file count (split frontend/backend)
- File types and distribution
- Lines of code per file
- Directory structure map

### Step 3: Test Discovery & Classification

Identify test files using path patterns, naming conventions, and import analysis.

| Test Type | Detection Heuristics |
|-----------|---------------------|
| Unit | `*Test.java`, `*_test.go`, `*_test.rs`, `test_*.py`, `*.test.ts`, `*.spec.ts` — no external service imports, no Spring context, no DB connections |
| Integration | `*IT.java`, `*Integration*`, `@SpringBootTest`, `@Testcontainers`, files importing DB/HTTP clients in test dirs |
| Contract | `*Contract*`, `*Pact*`, pact dependencies, Spring Cloud Contract |
| E2E | Files importing `@playwright/test`, `playwright` in test dirs, `*.e2e.ts`, `*.e2e.spec.ts` |
| CSS | `*.css.test.*`, visual regression tests, files importing `jest-image-snapshot`, `percy`, `chromatic` |
| Stress | `*.k6.js`, `*.k6.ts`, files importing `k6/http`, k6 scripts |
| Chaos | Files importing chaos libraries (`chaos-monkey`, `litmus`), `*chaos*` in test dirs |
| Mutation | `pitest` config in pom.xml, `mutmut` config, `cargo-mutants` config, mutation report files |
| Observability | Tests verifying logging output, metrics emission (`MeterRegistry`, `prometheus`), tracing (`opentelemetry`, `tracing`), health endpoints, alert rules |

### Step 4: Test Execution (SEQUENTIAL — single agent)

Before execution, the skill checks `metrics-config.json` for which test types to run. On first run, the skill generates a default config:

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

Mutation, stress, chaos, and CSS are disabled by default (slow/destructive). The user toggles them in the config.

**CRITICAL: All test execution happens in a single agent, sequentially, one test type at a time. Tests are NEVER run in parallel. The execution order is:**

1. **Backend tests first** (one at a time, wait for completion before next):
   - Unit tests
   - Integration tests
   - Contract tests
   - Observability tests

2. **Frontend tests second** (one at a time, wait for completion before next):
   - Unit tests
   - E2E tests (Playwright)
   - CSS visual tests

3. **Other test types last** (one at a time):
   - Stress tests (k6)
   - Chaos tests
   - Mutation tests

Execution commands per stack:

| Stack | Unit/Integration | E2E | Stress |
|-------|-----------------|-----|--------|
| Java (Maven) | `./mvnw test` / `./mvnw verify` | N/A (frontend handles) | `k6 run` |
| Java (Gradle) | `./gradlew test` / `./gradlew integrationTest` | N/A | `k6 run` |
| React/Node (npm) | `npm test` | `npx playwright test` | `k6 run` |
| React/Node (bun) | `bun test` | `npx playwright test` | `k6 run` |
| Python/Django | `pytest` / `python manage.py test` | `npx playwright test` | `k6 run` |
| Rust | `cargo test` | `npx playwright test` | `k6 run` |

Each execution captures:
- Exit code (pass/fail)
- stdout/stderr (for failure details)
- Duration
- Individual test results (parsed from test output / report files)

### Step 5: Coverage Collection (Hybrid)

#### Tool-Based Coverage (line-level precision)

| Stack | Tool | Config |
|-------|------|--------|
| Java (Maven) | JaCoCo | `jacoco-maven-plugin` |
| Java (Gradle) | JaCoCo | `jacoco` plugin |
| React/Node | Istanbul/c8 | `--coverage` flag |
| Python | coverage.py | `pytest --cov` |
| Rust | tarpaulin | `cargo tarpaulin` |

The skill checks if coverage tooling is already configured. If not, it runs tests with coverage flags where possible.

#### LLM-Based Coverage (all test types)

For every test file, the LLM reads the test code and traces which source files and functions are exercised:

- E2E test calls `page.goto('/api/users')` → maps to `UserController`, `UserService`, `UserRepository`
- k6 script hits `http.get('http://localhost:8080/health')` → maps to `HealthController`
- Contract test defines pact for `OrderService` → maps to `OrderController`, `OrderService`
- Observability test asserts `MeterRegistry` counter → maps to the service emitting that metric

Output per source file:

```json
{
  "file": "src/main/java/com/app/UserService.java",
  "layer": "backend",
  "coverage": {
    "unit": { "tool": 85.2, "llm": true },
    "integration": { "tool": 72.0, "llm": true },
    "e2e": { "tool": null, "llm": true },
    "stress": { "tool": null, "llm": true },
    "contract": { "tool": null, "llm": false },
    "chaos": { "tool": null, "llm": false },
    "mutation": { "tool": null, "llm": false },
    "observability": { "tool": null, "llm": false },
    "css": { "tool": null, "llm": false }
  }
}
```

Where `tool` = percentage from coverage tool (null if not applicable), `llm` = boolean if the LLM determined this file is exercised by that test type.

### Step 6: Git Attribution

For each test file, run `git log --format='%an' <file>` to get the author. Use `git blame` to attribute individual test methods to specific authors.

Output:
```json
{
  "authors": {
    "diegopacheco": {
      "unit": 42,
      "integration": 15,
      "e2e": 8,
      "stress": 3,
      "contract": 2,
      "chaos": 0,
      "mutation": 0,
      "observability": 5,
      "css": 0,
      "total": 75
    }
  }
}
```

### Step 7: LLM Quality Evaluation

For each test type, the LLM reads a sample of test files and evaluates:

- Are assertions meaningful (not just `assertNotNull`)?
- Do tests cover edge cases (nulls, empty, boundaries)?
- Are test names descriptive?
- Is there proper setup/teardown?
- Do integration tests actually test integration points?
- Do e2e tests cover critical user flows?
- Are stress tests hitting realistic load patterns?
- Do observability tests verify structured logging, metric names, trace propagation?

Each test type gets a quality rating: `poor`, `fair`, `good`, `excellent` with a short justification.

### Step 8: Score Computation (0-10)

| Criteria | Max Points | How |
|----------|-----------|-----|
| Test coverage breadth | 3 | % of source files covered by at least one test type |
| Test type diversity | 2 | How many different test types exist (out of 9). All 9 = 2pts, 5+ = 1.5, 3+ = 1, <3 = 0.5 |
| Pass rate | 2 | % of tests passing across all types |
| Test quality | 2 | Average of LLM quality ratings converted to numeric (poor=0, fair=0.5, good=1.5, excellent=2) |
| Code-to-test ratio | 1 | Ratio of test LOC to source LOC. >0.8 = 1pt, >0.5 = 0.7, >0.3 = 0.4, <0.3 = 0.1 |

### Step 9: JSON Output

All data is written to `metrics-report/data/metrics-latest.json` with the full structure:

```json
{
  "timestamp": "2026-03-31T14:30:00Z",
  "repository": {
    "name": "my-app",
    "url": "https://github.com/user/my-app",
    "branch": "main",
    "commit": "abc123"
  },
  "stacks": ["java-spring-boot", "react-node"],
  "files": {
    "total": 342,
    "backend": 210,
    "frontend": 132,
    "testFiles": 87,
    "byExtension": {}
  },
  "tests": {
    "total": 456,
    "passing": 440,
    "failing": 16,
    "byType": {
      "unit": { "total": 300, "passing": 295, "failing": 5, "files": [], "duration": 12.3 },
      "integration": { "total": 80, "passing": 75, "failing": 5, "files": [], "duration": 45.1 },
      "e2e": { "total": 30, "passing": 28, "failing": 2, "files": [], "duration": 120.5 },
      "stress": { "total": 5, "passing": 5, "failing": 0, "files": [], "duration": 60.0 },
      "contract": { "total": 10, "passing": 10, "failing": 0, "files": [], "duration": 8.2 },
      "chaos": { "total": 0, "passing": 0, "failing": 0, "files": [], "duration": 0 },
      "mutation": { "total": 0, "passing": 0, "failing": 0, "files": [], "duration": 0 },
      "observability": { "total": 20, "passing": 18, "failing": 2, "files": [], "duration": 5.0 },
      "css": { "total": 11, "passing": 9, "failing": 2, "files": [], "duration": 15.0 }
    }
  },
  "failures": {
    "byType": {},
    "details": [
      {
        "test": "UserServiceTest.testCreateUser",
        "type": "unit",
        "file": "src/test/java/com/app/UserServiceTest.java",
        "line": 45,
        "error": "Expected 200 but got 500",
        "githubUrl": "https://github.com/user/my-app/blob/main/src/test/java/com/app/UserServiceTest.java#L45"
      }
    ]
  },
  "coverage": {
    "backend": [],
    "frontend": []
  },
  "authors": {},
  "quality": {
    "byType": {
      "unit": { "rating": "good", "justification": "..." },
      "integration": { "rating": "fair", "justification": "..." }
    }
  },
  "score": {
    "total": 7.2,
    "breakdown": {
      "coverageBreadth": 2.4,
      "typeDiversity": 1.5,
      "passRate": 1.8,
      "testQuality": 1.0,
      "codeToTestRatio": 0.5
    }
  }
}
```

### Step 10: History Snapshot

After generating `metrics-latest.json`, copy it to `metrics-report/data/history/metrics-YYYY-MM-DDTHH-mm-ss.json`. The metrics application reads all history files to build trend charts.

### Step 11: Metrics Application Bootstrap

Only on first run (if `metrics-report/metrics-application/` does not exist), scaffold the React/Node/TypeScript application.

## Metrics Application

### Tech Stack

- React 19 + TypeScript
- Vite (build tool)
- Recharts (charting library)
- Node.js runtime
- No backend server needed — reads JSON files from `../data/` via Vite public dir or static serving

### Pages / Tabs

#### 1. Dashboard (Home)

- Overall score gauge (0-10) with color coding
- Total files, test files, tests count
- Pass/fail ratio donut chart
- Tests by type bar chart
- Current timestamp
- Stack badges detected
- Quick summary of quality evaluations

#### 2. Tests

- Table of all test files with columns: file, type, test count, pass/fail, author, quality rating
- Expandable rows showing individual test cases
- Clicking a test file → opens on GitHub
- Filter/search by name, type, author, status
- Color-coded pass/fail badges

#### 3. Coverage

- Split view: Backend tab / Frontend tab
- Table per file: file name, unit %, integration (mapped), e2e (mapped), stress (mapped), contract (mapped), observability (mapped)
- Color gradient cells (red 0% → green 100%, blue for "mapped")
- Searchable by file name
- Clicking a file → opens on GitHub

#### 4. Failures

- List of all failing tests grouped by type
- Each failure shows: test name, file, error message, stack trace
- Clicking a failure → opens the exact line on GitHub
- Historical failure frequency (how many times this test failed across runs)

#### 5. Authors

- Leaderboard of GitHub users by test count
- Breakdown per test type per author
- Bar charts comparing authors

#### 6. Trends

- Line chart: total tests over time
- Line chart: pass rate over time
- Line chart: score over time
- Stacked area chart: tests by type over time
- Line chart: coverage % over time (tool-based)
- Bar chart: failures per type over time
- All charts are interactive (hover for details, zoom, pan)
- Date range selector

#### 7. Quality

- Per test-type quality cards with rating badge and justification
- Score breakdown radar chart
- Recommendations for improvement

### Global Features

- Global search bar in the top nav — searches across all tabs (tests, files, authors, metrics)
- Clicking any test → GitHub link to file
- Clicking any metric value → opens the JSON file that contains that metric
- Dark/light theme toggle
- Responsive layout

### run.sh

```bash
#!/bin/bash
cd metrics-report/metrics-application
npm install
npm run build
npx serve -s dist -l 3737 &
echo "Metrics application running on http://localhost:3737"
```

### stop.sh

```bash
#!/bin/bash
pkill -f "serve -s dist -l 3737"
echo "Metrics application stopped"
```

## Progress Reporting (MANDATORY)

The skill MUST print progress at every phase transition and every test type execution. Format:

- Phase transitions: `[Phase X/6] <phase name>...` at start, `[Phase X/6] Done. <summary>` at end
- Test steps inside Agent 3: `[Test X/Y] Running <type> tests (<backend|frontend|other>)...` then `[Test X/Y] <type> tests done: <N> passed, <M> failed (<duration>s)`
- Final summary block with score, test counts, coverage, and stacks

The user must always know what is happening. Never go silent for more than one tool call without printing status.

## Skill Execution Flow

```
User invokes skill
       │
       ▼
[Phase 1/6] Setup and Detection
  Read metrics-config.json (create default if missing)
  Detect stacks present in codebase
       │
       ▼
[Phase 2/6] Parallel Scan (3 agents launched simultaneously)
  ┌──────────────────┬──────────────────┬──────────────────────────────┐
  │ Agent 1           │ Agent 2          │ Agent 3                      │
  │ File Count &      │ Git Attribution  │ Run Tests (SEQUENTIAL)       │
  │ Test Discovery    │ (batch)          │ One type at a time:          │
  │                   │                  │  1. Backend unit             │
  │ Glob/Grep to      │ Single bash cmd  │  2. Backend integration      │
  │ count files,      │ to get all test  │  3. Backend contract         │
  │ find tests,       │ file authors     │  4. Backend observability    │
  │ classify types    │                  │  5. Frontend unit            │
  │                   │                  │  6. Frontend e2e             │
  │                   │                  │  7. Frontend css             │
  │                   │                  │  8. Stress/Chaos/Mutation    │
  │                   │                  │                              │
  │                   │                  │ Reports [Test X/Y] progress  │
  └──────────────────┴──────────────────┴──────────────────────────────┘
       │
       ▼
[Phase 3/6] LLM Coverage Mapping
  Grep imports from test files → map to source files
       │
       ▼
[Phase 4/6] Quality Evaluation
  Read 3 representative test files per type → rate quality
       │
       ▼
[Phase 5/6] Compute Score and Generate JSON
  Score 0-10 → write metrics-latest.json
       │
       ▼
[Phase 6/6] History and Finalize
  Save history snapshot, copy data to metrics-application
       │
       ▼
Print final summary:
  === METRICS REPORT COMPLETE ===
  Score, tests, coverage, stacks
       │
       ▼
Auto-run: metrics-report/run.sh
  Metrics report live at http://localhost:3737
```

## File Structure

```
metrics-report/
├── data/
│   ├── metrics-latest.json
│   └── history/
│       ├── metrics-2026-03-31T14-30-00.json
│       └── metrics-2026-03-28T10-00-00.json
├── metrics-application/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── types/
│   │   │   └── metrics.ts
│   │   ├── components/
│   │   │   ├── SearchBar.tsx
│   │   │   ├── ScoreGauge.tsx
│   │   │   ├── TestTable.tsx
│   │   │   ├── CoverageTable.tsx
│   │   │   ├── FailureList.tsx
│   │   │   ├── AuthorLeaderboard.tsx
│   │   │   ├── QualityCard.tsx
│   │   │   └── charts/
│   │   │       ├── TrendLineChart.tsx
│   │   │       ├── TypeBarChart.tsx
│   │   │       ├── PassFailDonut.tsx
│   │   │       ├── ScoreRadar.tsx
│   │   │       └── StackedAreaChart.tsx
│   │   └── pages/
│   │       ├── Dashboard.tsx
│   │       ├── Tests.tsx
│   │       ├── Coverage.tsx
│   │       ├── Failures.tsx
│   │       ├── Authors.tsx
│   │       ├── Trends.tsx
│   │       └── Quality.tsx
│   └── public/
├── metrics-config.json
├── run.sh
└── stop.sh
```

## Edge Cases

- Monorepo with multiple stacks: scan each independently, merge results, tag each file/test with its stack
- No tests found: report score 0, empty tables, clear message
- Build fails before tests can run: capture the build error, mark all tests as "not executed", explain in failures tab
- No git remote: disable GitHub links, show local file paths instead
- Coverage tool not installed: skip tool-based coverage, rely on LLM-based mapping only, note it in the report
- History folder grows large: the app loads only the last 50 snapshots for trends
