# Decision Desk: Spring AI 2.1 Decision Models on a local model

A Spring Boot service and a React UI for the Spring AI 2.1 `DecisionModel` API, running against a **local Ollama decision model** (`d1-omni:600m`, no API key).
It is built on the Spring blog post [Decision Models in Spring AI 2.1: Yes, No, and How Sure](https://spring.io/blog/2026/11/26/spring-ai-decision-models-guardrails-judges).
You ask questions with fixed outcomes and get back typed answers (a boolean, an option, a level) with probabilities. You don't parse chat text.

## How it Works?

1. The UI posts a text to the service: a customer message, a text to screen, or a question and answer pair.
2. **Triage** builds one `DecisionPrompt` with three questions (`YesNoQuestion` defect, `ChoiceQuestion` topic, `ScoreQuestion` satisfaction) and calls the `DecisionModel` bean once.
3. The service routes on the **probability**, not the boolean: >= 0.9 opens a repair ticket automatically, 0.4 to 0.9 goes to a person, below 0.4 needs no action.
4. **Guardrail** runs the default input or output hazard battery of Spring AI Decision Extensions (`DecisionGuardrail`). All hazards are answered in one call, with PASS, REVIEW, BLOCK or SUPPORT as the result.
5. **Judge** runs a `DecisionJudge` with two criteria (helpfulness score, numeric plausibility yes/no), each with its own threshold. On failure it returns feedback a chat model could use for a second try.
6. The Ollama starter auto-configures the `DecisionModel` with `spring.ai.model.decision=ollama`. The same code runs on OpenAI or System One by switching the starter and that property.

## Architecture

![Architecture](printscreens/architecture.png)

## Features

* **One call, three typed answers**: defect, topic and satisfaction come from a single decision call, so the prompt needs no template, no JSON schema and no parser.
* **Probability based routing**: a "no" at 0.45 goes to human review, because the boolean alone hides a coin flip.
* **Confidence line**: the UI draws the yes probability on a 0 to 1 line against the backend thresholds, so you can see which "no" you can act on.
* **Input and output guardrails**: you can screen what the user sent or what a model answered, see every hazard score, and see the severity from 0 to 3.
* **Judge with feedback**: each criterion shows its own number and threshold, and a failure produces feedback text a self-refine loop can use.
* **Latency shown**: every result shows the model and the milliseconds of its single decision call (tens of ms once the model is warm).
* **Correlation ids**: the browser sends `X-Correlation-Id` on every call and the service logs it with path, user and service name.

## Stack

* **Java 25 + Spring Boot 4.2.0**: the service, with virtual threads enabled.
* **Spring AI 2.1.0** (`spring-ai-starter-model-ollama`): provides the `DecisionModel` API and its Ollama auto-configuration.
* **Spring AI Decision Extensions 0.5.0** (`spring-ai-decision-guardrail`, `spring-ai-decision-judge`): community guardrail batteries and the judge.
* **Ollama + `d1-omni:600m`**: a small local decision model; no API key, nothing leaves the machine.
* **Bun 1.4 + Vite 8 + React 19.3 + TypeScript 7**: the UI, built and tested with Bun.
* **TanStack Router, Query and Form**: file based tabs, mutations for the decision calls, and validated forms.

## Contracts / APIs

All endpoints take and return JSON. Validation errors return `400` and a failed model call returns `502`, both as `ProblemDetail`.

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/triage` | `{"message": "..."}` | `model`, `latencyMs`, `defect{yes, probability}`, `topic{value, probabilities}`, `satisfaction{label, nearestLevel, expectedLevel, levels, probabilities}`, `route`, `thresholds{auto, review}` |
| POST | `/api/guardrail` | `{"battery": "INPUT" \| "OUTPUT", "text": "..."}` | `outcome`, `blocked`, `severity`, `scores`, `triggered`, `flagged`, `summary`, `latencyMs` |
| POST | `/api/judge` | `{"question": "...", "answer": "..."}` | `passed`, `findings[{name, outcome, measure, value, threshold, detail}]`, `feedback`, `latencyMs` |

```bash
curl -s -X POST localhost:8097/api/triage -H 'Content-Type: application/json' \
  -d '{"message":"Where is my order? I ordered it two weeks ago."}'
```

## Key data structures and design decisions

* **Questions are typed, answers are typed**: `YesNoQuestion` -> `YesNoAnswer.getProbability()`, `ChoiceQuestion` -> `ChoiceAnswer.getProbabilities()`, `ScoreQuestion` -> `ScoreAnswer.getValue()` (expected level) and `getNearestLevel()`.
* **`Route` enum owns the routing rule** (`Route.of(probability, auto, review)`). Thresholds live in one `@ConfigurationProperties` record (`decisions.*`) and are sent to the UI, so the UI never keeps its own copy.
* **Judge score findings show probability mass**: Decision Extensions 0.5.0 passes a score criterion when the probability of the levels at or above `ceil(minimum)` is at least 0.5, not when the expected level reaches the minimum. The UI shows that mass next to the expected level, so a 1.81 expected level that PASSED makes sense.
* **Thresholds belong to a model**: the 0.9 / 0.4 / 0.5 values are tuned for `d1-omni:600m`. Re-validate them if you change the model.
* **No database**: nothing needs to be stored, so the stack's Postgres layer is left out.
* **Tests use a fake `DecisionModel`** that answers each question by its type, plus one live Spring Boot test that runs the real Ollama model end to end.

## How to run

Requires Java 25, Maven, Bun and Ollama (0.40.4 or later).

```bash
./scripts/setup.sh
./scripts/start-all.sh
./scripts/ui.sh
./scripts/test-all.sh
./scripts/stop-all.sh
```

`setup.sh` pulls `d1-omni:600m` when it is missing. `test-all.sh` runs 14 backend tests (3 of them live against the local model) and 9 frontend tests, after a production build and a type check.

## Printscreens

### Triage: a sure defect
![Triage defect](printscreens/01-triage-defect.png)
The headphones message: defect yes at 0.95, topic `product`, satisfaction `unhappy`. 0.95 is in the auto zone, so the route is an automatic repair ticket.

### Triage: a "no" that is not sure
![Triage review](printscreens/02-triage-review.png)
"Where is my order?": the boolean says no defect, but the probability is 0.45. That falls in the review band, so a person reads it. This is the blog's main point, and the confidence line makes it visible.

### Guardrail: prompt injection blocked
![Guardrail block](printscreens/03-guardrail-block.png)
The default input battery scores four hazards in one call. Jailbreak is at 0.79, so the outcome is BLOCK before any chat model would run.

### Guardrail: screening a model answer
![Guardrail output](printscreens/04-guardrail-output.png)
The output battery checks what a model answered, including whether it complied with something it should have refused.

### Judge: implausible answer fails
![Judge failed](printscreens/05-judge-failed.png)
-255°C in Paris: `is_plausible` scores 0.21 against 0.5 and fails. The feedback box shows the text a self-refine advisor would send back to the chat model.

### Judge: plausible answer passes
![Judge passed](printscreens/06-judge-passed.png)
15°C passes both criteria, so the verdict is PASSED with no feedback.

## Scripts

All scripts live in `scripts/` and run from any directory of the repository.

| Script | What it does |
|---|---|
| `./scripts/setup.sh` | Installs dependencies, builds the service and pulls the decision model |
| `./scripts/start-all.sh` | Starts Ollama when it is down, then the backend and frontend, and prints the full link of each one |
| `./scripts/status.sh` | Shows every service port as UP or DOWN |
| `./scripts/test-all.sh` | Runs every test suite |
| `./scripts/ui.sh` | Opens the UI in the browser |
| `./scripts/stop-all.sh` | Stops the backend and frontend (Ollama is shared and left running) |

Ports are declared in `scripts/ports.env`: backend `8097` (`http://localhost:8097/api`), frontend `5197` (`http://localhost:5197`), ollama `11434`.

```bash
./scripts/setup.sh
./scripts/start-all.sh
./scripts/status.sh
./scripts/ui.sh
./scripts/stop-all.sh
```
