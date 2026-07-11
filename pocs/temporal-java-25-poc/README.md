# temporal-java-25-poc

Java 25, Maven, Spring Boot 4.1.0, Spring AI 2.x, Temporal, PostgreSQL, HikariCP, Spring Data JDBC, and Swagger.

The service runs a Temporal workflow with three agents:

1. Stock agent researches current stock signals through `codex exec`.
2. News agent researches latest company news through `codex exec`.
3. Decision agent judges `BUY` or `HOLD` through `codex exec`.

## Run

```bash
./start.sh
```

## Stop

```bash
./stop.sh
```

## Test

```bash
./test.sh
```

## Logs

The app uses Log4j2 through `src/main/resources/log4j2-spring.xml`.

Logs go to the console and to `logs/temporal-java-25-poc.log`.

The default INFO logs show workflow ID, run ID, task queue, activity names, activity attempts, Codex CLI start, Codex CLI wait, Codex CLI response, output length, decisions, and report persistence.

Use this to change the Codex CLI activity budget:

```bash
CODEX_TIMEOUT_SECONDS=20 ./start.sh
```

## Temporal UI

```bash
./temporal-ui.sh
```

Open `http://localhost:8081`.

Workflow timeline:

`http://localhost:8081/namespaces/default/workflows/company-research-AAPL-8746b14f-6d04-487e-a2d7-8c07cf51fbce/019f52dc-848b-79a7-91bb-24aa84cb3266/timeline`

![Temporal workflow timeline](docs/temporal-workflow.png)

## Swagger

Open `http://localhost:8082/swagger-ui` or `http://localhost:8082/swagger`.

## API

```bash
curl -s -X POST http://localhost:8082/api/research \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"AAPL","company":"Apple"}'
```

```bash
curl -s 'http://localhost:8082/api/research?page=0&size=10'
```

## Trigger Temporal From REST

```bash
./trigger-temporal-rest.sh
```

```bash
./trigger-temporal-rest.sh MSFT Microsoft
```

The script calls Spring Boot at `POST http://localhost:8082/api/research/trigger`. That endpoint starts a Temporal workflow and returns immediately with the workflow ID, run ID, and Temporal UI URL.

`POST http://localhost:8082/api/research` still runs the blocking path. It waits for the workflow result and saves the report in PostgreSQL.

## What is Temporal

Temporal is an open-source durable execution platform. You write your business process as ordinary code (a workflow), and Temporal makes that code survive process crashes, restarts, deploys, and machine failures without losing state. It came out of Uber's Cadence project and is maintained by Temporal Technologies.

### How it works

- You write a **workflow**: the durable plan, ordinary code that must be deterministic (no direct clock, random, network, or disk access).
- You write **activities**: the side-effecting steps (API calls, DB writes, a `codex exec` subprocess). Activities can fail, time out, and are retried.
- A **worker** process hosts your workflow and activity code and polls a **task queue** for work.
- The **Temporal server** persists an append-only **event history** for every workflow. It hands work to workers and records every scheduled task, timeout, retry, and result.
- Durability comes from **replay**: if a worker dies mid-workflow, another worker replays the event history to rebuild in-memory state exactly where it left off, then continues. This is why workflow code must be deterministic.
- Extras: per-activity **timeouts and retry policies**, durable **timers**, **signals** (send data into a running workflow), **queries** (read workflow state), and **schedules/crons**.

### Who uses it

Uber (Cadence, its predecessor), Netflix, Stripe, Snap, Datadog, Coinbase, Instacart, HashiCorp, Descript, Retool, and many others use Temporal or Cadence for orchestration.

### Use cases

- Order, payment, and financial transaction processing.
- Saga / distributed-transaction coordination with compensation on failure.
- Infrastructure and resource provisioning.
- ETL and data pipelines.
- Human-in-the-loop approval flows.
- AI agent orchestration and multi-step LLM pipelines (what this app does).
- Any long-running business process that must not lose state.

### Pros

- Durable state and automatic recovery across crashes, restarts, and deploys.
- Built-in retries, timeouts, and backoff instead of hand-rolled state machines.
- Full visibility: the event history and UI show every step, attempt, and failure.
- Failure handling and long-running timers are first-class.
- Business logic stays as readable code, not a pile of queues and cron jobs.

### Cons

- Extra infrastructure to run and operate: the Temporal server plus a database.
- Workflow code must be deterministic; the constraints (no direct time, random, or IO) take time to learn.
- Versioning long-running workflows across code changes is non-trivial.
- Event history has size limits, so very large loops need `continueAsNew`.
- Added conceptual overhead for simple, short-lived tasks that would not benefit from durability.

## Temporal Session

Temporal is the durable orchestration layer in this app. Spring Boot receives HTTP traffic, but Temporal owns the long-running stock research process. If the JVM, worker, network, or an agent call fails, Temporal keeps workflow history and can retry work from the last recorded event.

The request flow is:

1. `trigger-temporal-rest.sh` sends a stock symbol and company name to `ResearchController`.
2. `ResearchService` creates a `CompanyResearchWorkflow` stub using the Temporal Java SDK.
3. Temporal schedules the workflow on the `company-stock-research` task queue.
4. The Spring Boot worker registered in `TemporalConfig` polls that task queue.
5. `CompanyResearchWorkflowImpl` runs the agent pipeline.
6. `StockAgentActivityImpl` calls `CodexCliService` for stock research.
7. `NewsAgentActivityImpl` calls `CodexCliService` for latest news research.
8. `DecisionAgentActivityImpl` calls `CodexCliService` to choose `BUY` or `HOLD`.
9. `ResearchService` saves the final `ResearchReport` through Spring Data JDBC.
10. `GET /api/research?page=0&size=10` reads saved reports with repository pagination.

Temporal separates workflow code from activity code. The workflow is the durable plan. Activities are the side-effecting agent calls. This matters because `codex exec` can be slow or fail. Temporal records each activity attempt, timeout, retry, and final result in the workflow timeline. Each activity is capped at 3 attempts. Codex timeout, exit failure, or process failure is thrown as an activity error so Temporal retries the same activity instead of moving to the next agent with bad data.

Spring AI is only a thin prompt wrapper here, not a real integration. `CodexCliService` builds a `Prompt` with a `UserMessage` and immediately calls `getContents()` to get the same string back, then passes it to `codex exec`. There is no `ChatModel` or `ChatClient` wired up, so no model call goes through Spring AI. The only Spring AI dependency used is `spring-ai-model`, for the message POJOs. The actual model call is the Codex CLI subprocess. The CLI runs with `--ephemeral`, `--sandbox read-only`, `--color never`, and `--output-last-message`.

The Temporal UI screenshot above shows a workflow named `company-research-AAPL-8746b14f-6d04-487e-a2d7-8c07cf51fbce`. The timeline view shows workflow input, workflow type, task queue, event history, and the current activity. In the captured run, `ResearchStock` is visible in the timeline. If Codex times out, the activity fails and Temporal retries it. The app gives the CLI 120 seconds by default and the Temporal activity 150 seconds. The activity budget must stay larger than the CLI budget so the CLI's own timeout fires first instead of Temporal killing the activity mid-call.

## Codex exec traps

Running `codex exec` from a Java `ProcessBuilder` (or any parent process) has sharp edges. These are the must-do items to avoid hangs and flaky bugs, all learned the hard way in this app.

- **Close the child's stdin.** `codex exec` reads stdin when stdin is not a TTY and appends it as a `<stdin>` block. `ProcessBuilder` gives the child an open stdin pipe that is never written to and never closed, so codex blocks reading it forever and only dies when the timeout kills it. Right after `start()`, call `process.getOutputStream().close()` (or redirect input from `/dev/null`) to send immediate EOF. This was the root cause of every "Codex CLI timed out. killed=true" in this app.
- **Drain stdout concurrently.** codex streams a lot of output. If you call `readAllBytes()` only after `waitFor()`, and the output exceeds the OS pipe buffer (~64KB), codex blocks writing to a full pipe while you block waiting for exit: a classic deadlock that looks like a timeout. Read the stream on a separate thread while the process runs, or redirect output to a file.
- **Lower the reasoning effort for latency-sensitive calls.** The default reasoning effort is high; a real research prompt with web search took ~40s. `-c model_reasoning_effort=low` did the same live web research in ~16s and still returned parseable results. This app sets it through `CODEX_REASONING_EFFORT` (default `low`).
- **Budget timeouts realistically and layer them.** Web-search research is slow. Set a generous CLI timeout (`CODEX_TIMEOUT_SECONDS`, default 120) and keep the Temporal activity `startToCloseTimeout` larger (150) so the CLI's own timeout fires first.
- **Kill the process hard on timeout and confirm the kill.** On timeout, call `destroyForcibly()` then `waitFor` briefly and log whether it actually died, so a stuck child never lingers.
- **Run non-interactive and capture a clean result.** Use `--skip-git-repo-check`, `--ephemeral`, `--sandbox read-only`, `--color never`, and `--output-last-message <file>` so you read the final answer from a file instead of parsing the streamed log. Never rely on codex prompting for approval; there is no TTY to answer it.
- **Delete the temp output file.** `--output-last-message` writes a temp file per call; clean it up in a `finally` block to avoid leaking files.
