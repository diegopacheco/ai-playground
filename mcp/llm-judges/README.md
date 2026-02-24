# LLM Judges MCP

MCP server that sends content to 4 LLM judges (Claude, Codex, Copilot, Gemini) via CLI and returns a consolidated verdict.

## Install

```bash
./install.sh
```

This will:
1. Build the Rust binary
2. Register the MCP in Claude Code (user scope)
3. Register the MCP in Codex CLI

## Tools

### judge
Send content to all 4 judges for evaluation.
```
"fact check this: Rust was created by Mozilla in 2010"
```

### judge_pick
Send content to selected judges.
```
"judge this using claude and gemini: the code is thread-safe"
```

### list_judges
List available judges.

## Requirements

- Rust 1.93+
- Claude CLI (`claude`)
- Codex CLI (`codex`)
- Copilot CLI (`copilot`)
- Gemini CLI (`gemini`)

## Result

```
❯ can you judge this?

  ❯ is AI coding agents real productivity or pure bs/hype?

  ⏺ Good question. Here's my honest take:

    Real productivity gains exist, but they're uneven.

    Where agents genuinely help:
    - Boilerplate generation, scaffolding, repetitive patterns
    - Exploring unfamiliar codebases faster
    - Writing tests for existing code
    - Translating between languages/frameworks
    - Rubber-ducking and catching obvious bugs
    - Tasks with clear specs and well-defined boundaries

    Where the hype outpaces reality:
    - Complex architectural decisions still need human judgment
    - Agents struggle with large, interconnected codebases where context matters
    - They generate plausible-looking but subtly wrong code that passes a quick review
    - Debugging agent-generated code can take longer than writing it yourself
    - The "10x productivity" claims usually cherry-pick simple tasks
    - They amplify the skill of the developer using them -- a junior gets less value than a senior

    The uncomfortable truth:

    AI agents are a power tool, not a replacement for thinking. A chainsaw makes a skilled lumberjack faster. It doesn't make
  someone
     who's never felled a tree safe.

    The real risk is:
    1. Velocity illusion -- shipping more code faster isn't the same as shipping better software
    2. Understanding debt -- if you don't understand what was generated, you can't maintain it
    3. Complexity creep -- easy generation encourages over-engineering

    Bottom line: Real productivity for experienced developers who know what to ask for, know how to verify output, and use it
  as
    acceleration rather than replacement for understanding. Pure hype if you think it eliminates the need to actually know how
   to
    program.

    It's a tool. A good one. Not magic.

⏺ llm-judges - judge (MCP)(content: "Question: \"Is AI coding agents real productivity or pure bs/hype?\"\n\nAnswer:\n\"Good
                          question. Here's my honest take:\n\nReal productivity gains exist, but they're uneven.\n\nWhere agents
                          genuinely help:\n- Boilerplate generation, scaffolding, repetitive patterns\n- Exploring unfamiliar
                          codebases faster\n- Writing tests for existing code\n- Translating between languages/frameworks\n-
                          Rubber-ducking and catching obvious bugs\n- Tasks with clear specs and well-defined boundaries\n\nWhere
                          the hype outpaces reality:\n- Complex architectural decisions still need human judgment\n- Agents
                          struggle with large, interconnected codebases where context matters\n- They generate plausible-looking
                          but subtly wrong code that passes a quick review\n- Debugging agent-generated code can take longer than
                          writing it yourself\n- The \"10x productivity\" claims usually cherry-pick simple tasks\n- They amplify
                          the skill of the developer using them -- a junior gets less value than a senior\n\nThe uncomfortable
                          truth:\n\nAI agents are a power tool, not a replacement for thinking. A chainsaw makes a skilled
                          lumberjack faster. It doesn't make someone who's never felled a tree safe.\n\nThe real risk is:\n1.
                          Velocity illusion -- shipping more code faster isn't the same as shipping better software\n2.
                          Understanding debt -- if you don't understand what was generated, you can't maintain it\n3. Complexity
                          creep -- easy generation encourages over-engineering\n\nBottom line: Real productivity for experienced
                          developers who know what to ask for, know how to verify output, and use it as acceleration rather than
                          replacement for understanding. Pure hype if you think it eliminates the need to actually know how to
                          program.\n\nIt's a tool. A good one. Not magic.\"", criteria: "Evaluate for: 1) Factual accuracy based on
                           current industry evidence and research, 2) Balanced perspective (not biased toward hype or dismissal),
                          3) Missing important nuances or counterpoints, 4) Whether the claims are well-supported by real-world
                          evidence")
  ⎿  {
       "verdict": "SPLIT",
       "score": "2/4",
     … +31 lines (ctrl+o to expand)

⏺ Here's the judgment from the LLM panel:

  Verdict: SPLIT (2/4 judges responded, both PASS)

  Two judges timed out/errored, but the two that responded both passed it:

  Codex (PASS, medium confidence):
  Claims broadly match industry studies and practitioner reports. Balanced view with benefits + limits. Flags key failure modes
  like "velocity illusion" and "understanding debt." Could be stronger by noting security/privacy/IP risks and that measured
  productivity varies widely by task type, tooling, and team processes.

  Gemini (PASS, high confidence):
  Accurately reflects current industry data showing significant productivity gains in boilerplate and testing. Correctly identifies
   that complex architectural context and "hallucinated" bugs remain major hurdles. Maintains balanced perspective distinguishing
  tool-assisted acceleration from the dangerous "velocity illusion."

  What the judges say is missing:
  - Security, privacy, and IP/licensing risks of AI-generated code
  - Productivity variance by task type, tooling, and team maturity
  - Could cite specific studies (e.g., GitHub Copilot research showing ~55% faster for specific tasks, not across the board)

  Fair feedback. The answer is directionally correct and balanced, but could go deeper on the risk dimensions beyond just code
  quality.

✻ Crunched for 3m 17s
```