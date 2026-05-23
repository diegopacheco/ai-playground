# gstack

https://github.com/garrytan/gstack

## Install

```
git clone --single-branch --depth 1 https://github.com/garrytan/gstack.git ~/.claude/skills/gstack && cd ~/.claude/skills/gstack && ./setup
```

## Experience Notes

* Using claude code and opus 4.7 with 1M ctx window.
* Installation was easy and smoth
* /office-hours ask you a lot of questions
* The skill is way too big and blow the context window: "⚠ Large CLAUDE.md will impact performance (48.5k chars > 40.0k)"
* Just to desgin a md file took 14% of my subscription
* So the second command /plan-eng-review just keep trying to convince you to reduce scope
* /plan-eng-review was also pushing to consider SaaS to be used (Fly.io or Railway) - this is really a startup mentality.
* Skill sugest you run codex
* /design-shotgun it's interesting - it give you multiple options to chose.
* Skills are asking for open ai API KEY all the time - kind of anoying.
* Skill say why: 
```
⏺ Honest answer: Claude doesn't generate raster images. I can SEE images (vision), describe them, write code that renders images
  (HTML/SVG/canvas), but I can't synthesize a PNG/JPEG from a prompt the way DALL-E / Imagen / gpt-image-1 do. Anthropic has no public
  image-generation API. The gstack designer wraps OpenAI's image API, which is why it needs an OpenAI key.
```

# Design Preview

After the design skills wrapped, Claude wrote a static HTML preview of the **qa2pw**
playground at [`preview.html`](./preview.html) using the tokens locked into
[`DESIGN-SYSTEM.md`](./DESIGN-SYSTEM.md) — real fonts (Inter Tight + JetBrains Mono),
the amber accent, cardless surfaces, sharp radius, and the full state matrix from
the plan-design-review session. The screenshots below are captured states from
that page. Open `preview.html` in a browser to flip between them live.

## 1. Idle — first paint

![idle state](./preview/Screenshot%202026-05-23%20at%2011.52.40.png)

The hero-by-example pattern (decision **D2** from the plan-design-review). The
textarea ships prefilled with `Log in with standard_user / secret_sauce, see the
inventory page` and the URL is already pointed at saucedemo.com — which is on the
allowlist, so the attestation checkbox is hidden and the green `allowlisted` badge
shows instead. A visitor can read the page and click Generate inside three seconds
without needing instructions. The center pane runs the dashed-outline browser
chrome with the "Click Generate to watch Claude work." hint, and the script pane
is anchored by a single syntax-highlighted comment placeholder. Download is
disabled until a run completes.

## 2. Streaming — Claude drives the browser

![streaming state](./preview/Screenshot%202026-05-23%20at%2011.52.52.png)

Form locks, Generate button reads `Generating… (step 8 of 25)`, and the step
caption uses the **action + reason** format chosen in **D5**: "Clicking the
login button — because the prompt says 'log in with standard_user'." The fake
saucedemo login form fills in real time with an amber pulse ring around the
LOGIN button so the user knows what's about to be clicked. The script pane
streams Playwright lines in as each action lands — line 8 is dim because it's
mid-write. This is the 30-90 second window where most playgrounds feel broken;
the streaming UX makes it feel alive instead.

## 3. Complete — run finished, script ready

![complete state](./preview/Screenshot%202026-05-23%20at%2011.53.05.png)

Green check overlay sits top-right of the frozen last frame (the
saucedemo inventory page). The step caption flips to "Run complete — Saved as
login.spec.ts, ready to download." Generate button reverts to a primary state
labeled `Generate again`. Form re-enables. Download button (outline variant)
goes live in the script pane. Full 10-line Playwright test visible in the
custom amber-palette syntax theme — `await`/`import`/`from` in amber, strings
in warm brown, comments muted. The user sees a coherent, committable test
file, not an agent-y blob.

## 4. Partial — timeout with recovery

![partial timeout state](./preview/Screenshot%202026-05-23%20at%2011.53.18.png)

The state that the plan-design-review decision **D3** invented to turn a failure
mode into a feature. A more ambitious test (`Add 3 items to cart, check out as
standard_user, verify confirmation page shows order summary with the right total`)
hits the 25-step LLM budget at step 18. Instead of returning nothing, the script
pane shows what was written so far, headlined by an **amber banner** ("Stopped at
step 18 of 25 — step budget exhausted on the checkout flow…") with a
**Continue from step 18 →** button that primes a new run with the existing
action log. The Download button changes to `Download partial`. The frozen frame
shows the checkout overview page the run reached. Failure becomes resumable
progress instead of a dead end.

## 5. Sleeping page — daily budget hit

![sleeping page state](./preview/Screenshot%202026-05-23%20at%2011.53.29.png)

The full-page takeover from decision **D5**. When the global $20/day Anthropic
spend ceiling fires, `/api/generate` stops responding and the entire site
collapses to this calm, centered page: wordmark, `Playground's resting until
midnight UTC.`, one-line "Daily budget hit. Star the repo to get notified when
v2 lifts the cap.", a ghost button to GitHub. No marketing fluff, no clock
graphic, no apology theater. The rate-limit page itself doubles as a low-key
marketing surface — every viral spike turns into GitHub stars instead of a
surprise bill.

---

The top bar across every screenshot (`qa2pw preview —` with state toggles and
the "preview.html — not in product" note on the right) is preview chrome,
not part of the shipped playground. It exists so you can flip between states
without scrolling. When the real Next.js app gets built (tasks T6–T7 in
[`DESIGN.md`](./DESIGN.md)), this bar disappears.