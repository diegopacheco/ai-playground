import type {
  Action,
  ActionLogEntry,
  ActionResult,
  GenerateOptions,
  RunEvent,
  RunResult,
  StopReason,
} from "./types.ts";
import { DEFAULTS } from "./defaults.ts";
import type { BrowserAdapter } from "./browser.ts";
import { selectorLabel } from "./browser.ts";
import { LimitGuard } from "./limits.ts";
import {
  type OllamaClient,
  HttpOllamaClient,
  OllamaError,
} from "./ollama.ts";
import {
  ACTION_SCHEMA,
  SYSTEM_PROMPT,
  buildUserMessage,
  formatHistory,
} from "./prompts.ts";
import { parseAction } from "./parseAction.ts";

export interface RunDependencies {
  browser: BrowserAdapter;
  ollama?: OllamaClient;
  now?: () => number;
}

export async function runGenerate(
  options: GenerateOptions,
  deps: RunDependencies,
): Promise<RunResult> {
  const maxSteps = options.maxSteps ?? DEFAULTS.maxSteps;
  const wallClockMs = options.wallClockMs ?? DEFAULTS.wallClockMs;
  const ollamaUrl = options.ollamaUrl ?? DEFAULTS.ollamaUrl;
  const model = options.model ?? DEFAULTS.model;
  const ollama = deps.ollama ?? new HttpOllamaClient(ollamaUrl);
  const now = deps.now ?? Date.now;

  const guard = new LimitGuard({ maxSteps, wallClockMs, now });
  const log: ActionLogEntry[] = [];
  const startedAt = now();
  const emit = (event: RunEvent) => options.onEvent?.(event);

  emit({ type: "status", status: "started", timestamp: startedAt });

  try {
    await deps.browser.goto(options.url);
  } catch (e) {
    emit({
      type: "status",
      status: "error",
      detail: `failed to open ${options.url}: ${(e as Error).message}`,
      timestamp: now(),
    });
    return {
      url: options.url,
      prompt: options.prompt,
      log,
      stopReason: "browser_error",
      startedAt,
      endedAt: now(),
    };
  }

  let stopReason: StopReason = "model_error";

  while (true) {
    const peek = guard.peek();
    if (peek.kind === "trip") {
      stopReason = peek.reason;
      break;
    }

    const screenshot = await deps.browser.screenshot();
    const currentUrl = await deps.browser.currentUrl();
    const history = formatHistory(
      log.map((entry) => ({
        verb: actionVerb(entry.action),
        ok: entry.result.status === "ok",
        detail: actionDetail(entry.action),
      })),
    );

    let raw: string;
    try {
      const response = await ollama.chat({
        model,
        stream: false,
        format: ACTION_SCHEMA,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          {
            role: "user",
            content: buildUserMessage({
              prompt: options.prompt,
              currentUrl,
              step: guard.currentStep + 1,
              maxSteps,
              history,
            }),
            images: [screenshot],
          },
        ],
        options: { temperature: 0.2 },
      });
      raw = response.message.content;
    } catch (e) {
      if (e instanceof OllamaError || e instanceof Error) {
        emit({
          type: "status",
          status: "error",
          detail: `model error: ${e.message}`,
          timestamp: now(),
        });
      }
      stopReason = "model_error";
      break;
    }

    const parsed = parseAction(raw);
    if (parsed.ok === false) {
      emit({
        type: "status",
        status: "error",
        detail: `model returned invalid action: ${parsed.error}`,
        timestamp: now(),
      });
      stopReason = "model_error";
      break;
    }

    const verdict = guard.tick();
    if (verdict.kind === "trip") {
      stopReason = verdict.reason;
      break;
    }

    const stepStart = now();
    const action = parsed.action;
    emit({
      type: "step",
      step: guard.currentStep,
      verb: actionVerb(action),
      reason: action.reason,
      timestamp: stepStart,
    });

    const result = await executeAction(action, deps.browser, now);
    log.push({
      step: guard.currentStep,
      action,
      result,
      timestamp: stepStart,
    });

    if (action.tool === "done") {
      stopReason = "done";
      break;
    }
    if (result.status === "failed") {
      continue;
    }
  }

  const endedAt = now();
  emit({
    type: "status",
    status: stopReason === "done" ? "complete" : stopReason === "step_budget" || stopReason === "wall_clock" ? "partial" : "error",
    detail: stopReason,
    timestamp: endedAt,
  });

  return {
    url: options.url,
    prompt: options.prompt,
    log,
    stopReason,
    startedAt,
    endedAt,
  };
}

async function executeAction(
  action: Action,
  browser: BrowserAdapter,
  now: () => number,
): Promise<ActionResult> {
  const start = now();
  try {
    switch (action.tool) {
      case "click":
        await browser.click(action.selector);
        break;
      case "type":
        await browser.type(action.selector, action.text);
        break;
      case "wait_for":
        await browser.waitFor(action.selector);
        break;
      case "assert_text":
        await browser.assertText(action.selector, action.text);
        break;
      case "screenshot":
        await browser.screenshot();
        break;
      case "done":
        break;
    }
    return { status: "ok", action, tookMs: now() - start };
  } catch (e) {
    return {
      status: "failed",
      action,
      error: (e as Error).message,
      tookMs: now() - start,
    };
  }
}

function actionVerb(action: Action): string {
  switch (action.tool) {
    case "click":
      return `Clicking ${selectorLabel(action.selector)}`;
    case "type":
      return `Typing ${JSON.stringify(action.text)} into ${selectorLabel(action.selector)}`;
    case "wait_for":
      return `Waiting for ${selectorLabel(action.selector)}`;
    case "assert_text":
      return `Asserting ${selectorLabel(action.selector)} contains ${JSON.stringify(action.text)}`;
    case "screenshot":
      return "Taking a screenshot";
    case "done":
      return "Finishing the test";
  }
}

function actionDetail(action: Action): string {
  return action.reason;
}
