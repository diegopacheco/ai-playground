import { EVAL_CASES, type EvalCase } from "./cases.ts";
import {
  actionLogToScript,
  chromiumLauncher,
  DEFAULTS,
  HttpOllamaClient,
  PlaywrightSession,
  runGenerate,
  withDisposable,
} from "../src/index.ts";

interface EvalResult {
  case: EvalCase;
  status: "pass" | "fail" | "error" | "skip";
  stopReason?: string;
  steps?: number;
  durationMs?: number;
  reason?: string;
  script?: string;
  missingExpectations?: string[];
}

const BAR_PASS_RATIO = 0.8;

async function ollamaUp(url: string): Promise<boolean> {
  try {
    const r = await fetch(`${url}/api/tags`);
    return r.ok;
  } catch {
    return false;
  }
}

async function modelPresent(url: string, model: string): Promise<boolean> {
  try {
    const r = await fetch(`${url}/api/tags`);
    if (!r.ok) return false;
    const body = (await r.json()) as { models?: Array<{ name: string }> };
    return (body.models ?? []).some((m) => m.name === model);
  } catch {
    return false;
  }
}

async function runOne(c: EvalCase, model: string, ollamaUrl: string): Promise<EvalResult> {
  const start = Date.now();
  try {
    return await withDisposable(
      () => PlaywrightSession.launch(chromiumLauncher, { headless: true }),
      async (browser) => {
        const run = await runGenerate(
          {
            prompt: c.prompt,
            url: c.url,
            model,
            ollamaUrl,
            maxSteps: DEFAULTS.maxSteps,
            wallClockMs: DEFAULTS.wallClockMs,
          },
          { browser, ollama: new HttpOllamaClient(ollamaUrl) },
        );
        const script = actionLogToScript(run);
        const missing = c.expectInScript.filter((needle) => !script.includes(needle));
        const passed = run.stopReason === "done" && missing.length === 0;
        return {
          case: c,
          status: passed ? "pass" : "fail",
          stopReason: run.stopReason,
          steps: run.log.length,
          durationMs: Date.now() - start,
          script,
          missingExpectations: missing,
        };
      },
    );
  } catch (e) {
    return {
      case: c,
      status: "error",
      reason: e instanceof Error ? e.message : String(e),
      durationMs: Date.now() - start,
    };
  }
}

async function main(): Promise<void> {
  const ollamaUrl = process.env.QA2PW_OLLAMA_URL ?? DEFAULTS.ollamaUrl;
  const model = process.env.QA2PW_MODEL ?? DEFAULTS.model;
  const onlyId = process.env.EVAL_CASE_ID;

  if (!(await ollamaUp(ollamaUrl))) {
    console.error(`Ollama not reachable at ${ollamaUrl}. Run ./run.sh first.`);
    process.exit(2);
  }
  if (!(await modelPresent(ollamaUrl, model))) {
    console.error(`Model ${model} not pulled. Run: ollama pull ${model}`);
    process.exit(2);
  }

  const cases = onlyId
    ? EVAL_CASES.filter((c) => c.id === onlyId)
    : EVAL_CASES;
  if (cases.length === 0) {
    console.error(`No matching cases.`);
    process.exit(2);
  }

  console.log(`Running ${cases.length} eval(s) against ${model} via ${ollamaUrl}`);
  console.log("");
  const results: EvalResult[] = [];
  for (const c of cases) {
    process.stdout.write(`[${c.id}] `);
    const r = await runOne(c, model, ollamaUrl);
    results.push(r);
    const dur = r.durationMs !== undefined ? `${(r.durationMs / 1000).toFixed(1)}s` : "?";
    if (r.status === "pass") {
      console.log(`PASS in ${dur} (${r.steps} steps)`);
    } else if (r.status === "fail") {
      console.log(
        `FAIL in ${dur} (stop=${r.stopReason}, missing=[${r.missingExpectations?.join(", ")}])`,
      );
    } else if (r.status === "error") {
      console.log(`ERROR in ${dur}: ${r.reason}`);
    } else {
      console.log("SKIP");
    }
  }

  const passes = results.filter((r) => r.status === "pass").length;
  const total = results.length;
  const ratio = total > 0 ? passes / total : 0;
  console.log("");
  console.log(`Summary: ${passes}/${total} passed (${(ratio * 100).toFixed(0)}%). Bar: ${BAR_PASS_RATIO * 100}%.`);
  process.exit(ratio >= BAR_PASS_RATIO ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
