export type {
  Action,
  ActionLogEntry,
  ActionResult,
  FrameEvent,
  GenerateOptions,
  GenerateResult,
  RunEvent,
  RunResult,
  Selector,
  StatusEvent,
  StepEvent,
  StopReason,
  ToolName,
} from "./types.ts";

export type { BrowserAdapter } from "./browser.ts";
export type { OllamaClient } from "./ollama.ts";
export { HttpOllamaClient, OllamaError } from "./ollama.ts";
export { LimitGuard } from "./limits.ts";
export { runGenerate, type RunDependencies } from "./generate.ts";
export { parseAction, type ParseResult } from "./parseAction.ts";
export { actionLogToScript } from "./templater.ts";
export { checkUrl, ALLOWLIST, SAFETY_BLOCKLIST, type UrlCheck } from "./allowlist.ts";
export { withDisposable, type AsyncDisposable } from "./lifecycle.ts";
export {
  PlaywrightSession,
  withPlaywrightSession,
  type BrowserLauncher,
  type PlaywrightLaunchOptions,
} from "./playwrightAdapter.ts";
export { chromiumLauncher } from "./chromium.ts";
export { DEFAULTS } from "./defaults.ts";

export const RUNNER_VERSION = "0.0.1";

import type { GenerateOptions, GenerateResult } from "./types.ts";
import { runGenerate, type RunDependencies } from "./generate.ts";
import { actionLogToScript } from "./templater.ts";

export async function generateScript(
  options: GenerateOptions,
  deps: RunDependencies,
): Promise<GenerateResult> {
  const run = await runGenerate(options, deps);
  const script = actionLogToScript(run);
  return { script, run };
}
