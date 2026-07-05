# Hats

Hats is a Claude Code and Codex skill that applies Edward de Bono's Six Thinking Hats to a prompt and writes a polished, self-contained HTML report.

Every report includes the original prompt, six inline hat drawings, 3–6 focused insights per hat, and a blue-hat synthesis. Reports work offline and contain no external assets or libraries.

## Hats

- White captures facts, evidence, unknowns, and useful measures.
- Red captures instincts, emotions, tensions, and likely reactions.
- Black captures risks, constraints, and failure modes.
- Yellow captures benefits, opportunities, and reasons for confidence.
- Green captures alternatives, experiments, and new directions.
- Blue turns the analysis into priorities and next actions.

## Install

```bash
./install.sh
```

Choose Claude Code, Codex, or both. The installer uses these global locations:

```text
~/.claude/skills/hats
${CODEX_HOME:-~/.codex}/skills/hats
```

## Use

Claude Code:

```text
/hats Should we replace our weekly production release with continuous delivery?
```

Codex:

```text
$hats Should we replace our weekly production release with continuous delivery?
```

The skill names the result after the prompt's central intention, such as `adopt-continuous-delivery.html`. If that file exists, it adds a numeric suffix rather than replacing it.

## Included Report

Open [improve-deployment-reliability.html](improve-deployment-reliability.html) in any browser to inspect a completed report.

## Uninstall

```bash
./uninstall.sh
```

Choose Claude Code, Codex, or both.
