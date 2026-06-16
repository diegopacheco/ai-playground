---
name: bs-claudemd
description: Audit a CLAUDE.md for cruft and render a light-theme report website. Reads the file line by line and flags vague rules with no checkable criterion, low-signal lines that only restate what the model already does, overlapping rules said more than once, and internal contradictions, then scores the file with a BS-o-meter and grade. Can also diff two CLAUDE.md files and flag rules that tell the model opposite things. Use when the user runs /bs-claudemd or asks to review, audit, grade, lint, or clean up a CLAUDE.md / AGENTS.md / agent instructions file, or to compare two of them.
---

# bs-claudemd

Audit a CLAUDE.md file and build a self-contained report website that comments on every line.

## What it flags

- **Vague** — a rule with no checkable criterion ("as simple as possible", "well written", "make sense"); the model can't tell when it is satisfied.
- **Low signal** — a line that restates behavior the model already biases toward; it spends tokens without changing output.
- **Overlap** — the same instruction stated more than once, often across different sections; consolidate it.
- **Contradiction** — two rules whose plain reading conflicts, so the model cannot honor both.
- **Keepers** — concrete, checkable rules. The report keeps these visible so the user sees what is working.

Each line gets a one-line comment and a colored verdict. The header shows a BS-o-meter (0–100, lower is better), a letter grade, and metric cards including estimated tokens loaded every turn and estimated wasted tokens.

## Steps

1. Pick the target. With no argument, audit the global file at `~/.claude/CLAUDE.md`. If the user says "project" or names a path, use that file (a project `CLAUDE.md`, `AGENTS.md`, or any instruction file works). If the user gives two files, run diff mode.
2. Run the analyzer that ships with this skill (Python standard library only, nothing to install):

   ```
   python3 ~/.claude/skills/bs-claudemd/bs_claudemd.py analyze <claude.md> <out-dir> [title]
   ```

   For diff mode:

   ```
   python3 ~/.claude/skills/bs-claudemd/bs_claudemd.py diff <A.md> <B.md> <out-dir> [title]
   ```

   Use `bs-report` inside the current project as `<out-dir>` unless the user asks for another location. Pass a short title (the project name, or "global").
3. The script writes a self-contained `index.html` plus `data.json` and prints the headline: BS score, grade, and the counts. Serve the folder with `python3 -m http.server` from `<out-dir>` on a free port, or tell the user to open the file directly. Report the URL/path and the headline numbers.
4. Tell the user how to read it: the **Overview** tab has the verdict summary and rule-health donut; **Line-by-line** annotates every line with a filter and search box; **Overlaps** groups repeated rules; **Contradictions** lists conflicting rules; in diff mode a **Cross-file conflicts** tab shows where the two files tell the model opposite things.

## How verdicts are decided

- A line is treated as a rule if it is a bullet or carries a normative cue (never, always, must, prefer, don't, make sure, ...). Headings, blank lines, and fenced code are labeled but not scored.
- Overlap groups are rules that share a distinctive (rare) word and enough sentence overlap that they are plausibly the same instruction. Single shared common words are not enough.
- Contradictions bind each polarity cue (always / never / don't / must) to the words that follow it in the same clause, then look for the same distinctive word asserted both ways by rules that are otherwise about the same thing. This is deliberately strict, so a clean file reports zero.
- The BS score weights contradictions heaviest, then overlaps, then vague and low-signal lines, normalized by how many rules the file has.

## Notes

- The site is light themed with a hand-drawn accent (wobble-filtered SVG gauge, Caveat display font, pastel verdict colors). It loads a web font from Google Fonts and falls back to system fonts offline.
- The analysis is heuristic and opinionated. It flags candidates; the judgment stays with the user. "Low signal" and "vague" are suggestions to tighten wording, not orders to delete.
- It never edits the CLAUDE.md it audits. It only reads it and writes the report folder.
