export type ToolName =
  | "click"
  | "type"
  | "wait_for"
  | "assert_text"
  | "screenshot"
  | "done";

export type Action =
  | { tool: "click"; selector: Selector; reason: string }
  | { tool: "type"; selector: Selector; text: string; reason: string }
  | { tool: "wait_for"; selector: Selector; reason: string }
  | { tool: "assert_text"; selector: Selector; text: string; reason: string }
  | { tool: "screenshot"; reason: string }
  | { tool: "done"; reason: string };

export type Selector =
  | { kind: "role"; role: string; name?: string }
  | { kind: "placeholder"; text: string }
  | { kind: "text"; text: string }
  | { kind: "label"; text: string }
  | { kind: "test_id"; id: string };

export type ActionResult =
  | { status: "ok"; action: Action; tookMs: number }
  | { status: "failed"; action: Action; error: string; tookMs: number };

export interface ActionLogEntry {
  step: number;
  action: Action;
  result: ActionResult;
  screenshotPath?: string;
  timestamp: number;
}

export type StopReason =
  | "done"
  | "step_budget"
  | "wall_clock"
  | "model_error"
  | "browser_error"
  | "user_cancel";

export interface RunResult {
  url: string;
  prompt: string;
  log: ActionLogEntry[];
  stopReason: StopReason;
  startedAt: number;
  endedAt: number;
}

export type FrameEvent = {
  type: "frame";
  data: string;
  timestamp: number;
};

export type StepEvent = {
  type: "step";
  step: number;
  verb: string;
  reason: string;
  timestamp: number;
};

export type StatusEvent = {
  type: "status";
  status: "started" | "complete" | "error" | "partial";
  detail?: string;
  timestamp: number;
};

export type RunEvent = FrameEvent | StepEvent | StatusEvent;

export interface GenerateOptions {
  prompt: string;
  url: string;
  attested?: boolean;
  maxSteps?: number;
  wallClockMs?: number;
  ollamaUrl?: string;
  model?: string;
  onEvent?: (event: RunEvent) => void;
}

export interface GenerateResult {
  script: string;
  run: RunResult;
}
