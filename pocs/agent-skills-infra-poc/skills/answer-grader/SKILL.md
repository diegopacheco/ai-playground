---
name: answer-grader
description: Grade a written answer (release notes, summaries, commit messages, JSON payloads) against checks using the shared evaluators of the agent infra, store every run in the evals UI and track the score trend per suite. Use when the user says "grade this answer", "check this output", "does this meet the criteria", "score the release notes" or "show grading history".
---

# Answer Grader

Turns acceptance criteria into checks, grades an answer with the common evaluators of the
agent infra evals service, and remembers the score history per suite in this skill's own state
tenant (`answer-grader`). Every run shows up in the evals UI and in the centralized logs.

All commands run `scripts/grade.py` from this skill's base directory with Python 3.15 and print JSON.

## Commands

| Goal | Command |
|---|---|
| See the evaluators | `python3.15 scripts/grade.py evaluators` |
| Grade an answer | `python3.15 scripts/grade.py grade --suite <suite> --answer <file> --checks '<json list>'` |
| See the trend | `python3.15 scripts/grade.py history --suite <suite>` |

A check is `{"type": "<evaluator>", "value": <expected>}`, such as:

```json
[{"type": "contains", "value": "Breaking changes"}, {"type": "max_length", "value": 1200}, {"type": "not_contains", "value": "TODO"}]
```

## Rules

1. Write the answer to a file first, then grade the file.
2. Derive checks only from criteria the user gave. Do not invent criteria.
3. Use the same suite name for the same kind of answer so the trend is meaningful.
4. When `failed_checks` is not empty, fix the answer and grade again, at most 3 times.
5. Report the score, the trend and each failed check's `detail`.
