---
name: no-slop-pr
description: Scores GitHub pull requests for AI slop and blocks the ones an agent wrote, in the spirit of No Sloptober. Use when the user runs /no-slop-pr, or asks to check, gate, block, flag or reject AI generated, LLM written, agent authored or sloppy pull requests.
allowed-tools: [Bash, Read, AskUserQuestion]
---

# No Slop PR

No Sloptober asks for one month without LLM tools. This skill guards the other side of the door:
pull requests written by an agent do not get merged.

## Inputs

* A PR number or URL: `python3 .claude/skills/no-slop-pr/slop_check.py 42`
* A PR from another repo: `python3 .claude/skills/no-slop-pr/slop_check.py 42 --repo owner/name`
* All open PRs: loop over `gh pr list --state open --json number --jq '.[].number'`
* Local files, no GitHub: `python3 .claude/skills/no-slop-pr/slop_check.py --text body.txt --diff change.diff`

Add `--json` for machine output. The script exits `1` on `BLOCK` and `0` otherwise, so it can gate CI.

## Signals

Every signal, weight and threshold lives in `rules.json`. Read it, never guess.

| Signal | Why it matters |
|---|---|
| AI co-author trailer | The commit itself admits an agent wrote it |
| Generated-with footer | Same admission, in the PR body |
| LLM tell phrases | delve, robust, seamless, comprehensive, this PR introduces |
| Emoji bullets and headings | Checkmark and rocket bullet lists are an agent template |
| Em dash storm | Humans rarely type the long dash, models love it |
| Boilerplate section headings | Summary, Key Changes, Test Plan scaffolding |
| Comment cruft in added code | Onarheim: agents add comments that explain nothing |

Score is capped at 100. `BLOCK` at `block_at`, `SUSPECT` at `suspect_at`, `PASS` below.

## Steps

1. Run the script for every PR the user named.
2. Print one line per PR: number, verdict, score, the top reasons.
3. For `SUSPECT`, quote the lines that matched so a human decides. Never block on `SUSPECT` alone.
4. For `BLOCK`, ask the user before touching GitHub. Only after a yes run
   `gh pr review <n> --request-changes --body "<reasons>"`. Never close, merge or push.
5. The review body lists the reasons from the script and one line: `No Sloptober: please rewrite this PR yourself.`

## Rules

* The script is a filter, not a judge. Say so when reporting.
* Never write the review body with emojis, em dashes or any phrase from `rules.json`.
* Run the tests after changing `rules.json`: `python3 -m unittest discover -s .claude/skills/no-slop-pr/tests`.
