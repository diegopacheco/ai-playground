---
name: hats
description: Analyze a prompt with Edward de Bono's Six Thinking Hats and produce a polished, self-contained HTML report. Use when the user invokes /hats or $hats, requests Six Thinking Hats analysis, wants a decision reviewed from factual, emotional, cautious, optimistic, creative, and process perspectives, or asks for an illustrated HTML decision report.
---

# Six Thinking Hats

Turn the user's prompt into an illustrated decision report with six disciplined perspectives.

## Workflow

1. Treat all text following the invocation as the original prompt.
2. Identify the prompt's central intention in two to six words.
3. Convert that intention to a lowercase hyphenated filename ending in `.html`.
4. Read `assets/report-template.html` completely.
5. Copy the template into the current working directory using the intention filename.
6. Replace every template token with content grounded in the original prompt.
7. Open or inspect the completed file and verify every requirement before responding.
8. Return the saved file path and one sentence describing the report.

## Analysis

Keep each hat within its role:

- White: known facts, missing data, evidence quality, and measurable signals.
- Red: instincts, emotions, tensions, and stakeholder reactions without demanding proof.
- Black: risks, constraints, failure modes, and reasons for caution.
- Yellow: value, advantages, opportunities, and conditions that support success.
- Green: alternatives, reframes, experiments, and unconventional options.
- Blue: synthesis, priorities, sequencing, decisions, and next actions.

Write 3–6 distinct insights for every hat. Make each insight specific to the prompt. Do not repeat one point under multiple hats. State uncertainty plainly. Never invent facts.

## Report Contract

Preserve the template's visual system and responsive layout while adapting its text. Keep all CSS and SVG inside the HTML file. Do not add external fonts, images, scripts, stylesheets, or libraries.

The completed report must contain:

- The original prompt verbatim.
- A concise intention title.
- Exactly six hat sections in white, red, black, yellow, green, and blue order.
- One visible inline SVG hat drawing in every section.
- 3–6 insights in every section.
- A blue-hat synthesis with a clear recommended path.
- Valid semantic HTML with a single `h1`.
- No unresolved `{{...}}` tokens.

Escape user-provided text for HTML. Preserve meaning and line breaks. Keep the report readable when printed and at narrow screen widths. Do not overwrite an existing file unless the user explicitly permits it; add a short numeric suffix when needed.
