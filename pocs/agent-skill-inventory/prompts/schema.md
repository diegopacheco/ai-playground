---
name: schema
params: none
---

Contract for `analysis.json`. `render.py` validates against this and refuses to
build the report when it does not match.

```jsonc
{
  "project": {
    "name": "string",
    "summary": "1-3 lines, what this codebase does",
    "codebaseUrl": "absolute path on disk, from facts.repoRoot",
    "githubUrl": "https://github.com/org/repo or null"
  },

  "architecture": {
    "summary": "3-6 lines, how the parts fit together",
    "nodes": [
      { "id": "kebab-id", "label": "short name",
        "kind": "client|module|store|queue|external",
        "layer": 0 }
    ],
    "edges": [
      { "from": "node-id", "to": "node-id",
        "label": "verb phrase, 1-4 words",
        "evidence": "file path or config key that proves it" }
    ]
  },

  "modules": [
    { "id": "kebab-id",
      "name": "string",
      "icon": "one emoji",
      "description": "3 to 5 lines, plain language, no metaphors",
      "paths": ["dir/ relative to repo root"],
      "files": [ { "path": "full path from repo root", "role": "one line" } ],
      "pros": ["exactly 5 entries, one line each"],
      "cons": [
        { "title": "short name of the problem",
          "why":  "at most 2 lines, what goes wrong for a real person",
          "fix":  "at most 3 lines, a concrete action on named code" }
      ]
    }
  ],

  "schema": {
    "present": true,
    "engine": "postgres|mysql|sqlite|h2|mongo|none",
    "tableCount": 0,
    "sizeNote": "how big it is, or 'n/a - not reachable from a static scan'",
    "tables": [
      { "name": "table", "purpose": "one line", "columns": 0,
        "rows": "known row count or n/a", "size": "known size or n/a",
        "source": "path where it is defined" }
    ],
    "worstQueries": [
      { "query": "the SQL or JPQL, trimmed",
        "path": "file", "line": 0,
        "why": "at most 2 lines, why it is slow or risky",
        "fix": "at most 3 lines" }
    ],
    "pros": ["exactly 5"],
    "cons": [ { "title": "", "why": "", "fix": "" } ]
  },

  "committers": [
    { "name": "", "email": "", "login": "github login or null",
      "commits": 0, "firstCommit": "YYYY-MM-DD", "lastCommit": "YYYY-MM-DD",
      "areas": ["top directories this person touched"],
      "worksOn": "2-3 lines, what this person actually builds here",
      "profileUrl": "https://github.com/login or mailto: link" }
  ],

  "techDebt": [
    { "id": "kebab-id", "module": "module id",
      "title": "", "severity": "high|medium|low",
      "whyBad": "2-4 lines, the cost paid for keeping it",
      "howToFix": "3-5 lines, ordered steps on named code",
      "antiPatterns": [ { "name": "", "why": "1-2 lines" } ],
      "examples": [ { "path": "", "line": 0, "snippet": "the offending line" } ]
    }
  ],

  "observability": {
    "logsScore": 0,
    "logsVerdict": "2-4 lines",
    "logsFindings": [ { "title": "", "detail": "1-2 lines", "path": "" } ],
    "dashboards": [ { "name": "", "path": "", "type": "grafana|other" } ],
    "alerts": [ { "name": "", "path": "", "condition": "" } ],
    "defensive": {
      "score": 0,
      "verdict": "2-4 lines on input validation and error handling",
      "findings": [ { "title": "", "detail": "1-2 lines", "path": "" } ]
    }
  },

  "tests": {
    "summary": "2-4 lines",
    "types": [
      { "type": "unit|integration|e2e|stress|chaos|property|css|database|contract|smoke|benchmark|manual",
        "files": 0, "cases": 0, "verdict": "one line" }
    ],
    "coverage": [
      { "area": "what is being covered", "covered": 0, "total": 0 }
    ],
    "issues": [
      { "title": "", "why": "1-2 lines", "fix": "1-3 lines",
        "path": "", "line": 0 }
    ]
  },

  "verification": { "passes": 3, "notes": ["one line per correction"] }
}
```

## Hard rules

* `modules[].pros` has exactly 5 entries. `modules[].cons` has exactly 5.
* When `schema.present` is true, `schema.pros` has exactly 5 and
  `schema.cons` has exactly 5. When false, the whole tab is dropped and only
  `{"present": false}` is needed.
* `techDebt` carries at most 10 items per module id.
* `tests.issues` carries at most 10 items.
* Every `path` must exist on disk relative to the repo root.
* Every `architecture.edges[].from` and `.to` must be an existing node id.
* No field may contain a metaphor.
