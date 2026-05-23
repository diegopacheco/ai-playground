"use client";

import { useCallback, useRef, useState } from "react";

type StreamEvent =
  | {
      type: "status";
      status: "started" | "complete" | "error" | "partial";
      detail?: string;
      timestamp: number;
    }
  | { type: "step"; step: number; verb: string; reason: string; timestamp: number }
  | { type: "frame"; data: string; timestamp: number }
  | { type: "result"; script: string; stopReason: string; steps: number }
  | { type: "fatal"; message: string };

type Phase = "idle" | "streaming" | "complete" | "partial" | "error";

const DEFAULT_PROMPT =
  "Log in with standard_user / secret_sauce, see the inventory page";
const DEFAULT_URL = "https://www.saucedemo.com";

export function Playground() {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [url, setUrl] = useState(DEFAULT_URL);
  const [attested, setAttested] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [step, setStep] = useState<{ verb: string; reason: string } | null>(null);
  const [stepCount, setStepCount] = useState(0);
  const [script, setScript] = useState<string>("");
  const [stopReason, setStopReason] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [frame, setFrame] = useState<string>("");
  const abortRef = useRef<AbortController | null>(null);

  const isAllowlisted = /(^|\.)(saucedemo\.com|the-internet\.herokuapp\.com|automationexercise\.com|demo\.playwright\.dev|todomvc\.com)$/i.test(
    safeHost(url),
  );

  const onGenerate = useCallback(async () => {
    if (phase === "streaming") return;
    setPhase("streaming");
    setStep(null);
    setStepCount(0);
    setScript("");
    setStopReason("");
    setErrorMessage("");
    setFrame("");
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, url, attested }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const body = await response.text();
        setErrorMessage(`${response.status}: ${body}`);
        setPhase("error");
        return;
      }
      if (response.body === null) {
        setErrorMessage("server returned no body");
        setPhase("error");
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const messages = buffer.split("\n\n");
        buffer = messages.pop() ?? "";
        for (const block of messages) {
          const evt = parseSse(block);
          if (evt === null) continue;
          handleEvent(evt, {
            setStep,
            setStepCount,
            setScript,
            setStopReason,
            setPhase,
            setErrorMessage,
            setFrame,
          });
        }
      }
    } catch (e) {
      if (controller.signal.aborted) return;
      setErrorMessage(e instanceof Error ? e.message : String(e));
      setPhase("error");
    } finally {
      abortRef.current = null;
    }
  }, [prompt, url, attested, phase]);

  const onCancel = useCallback(() => {
    abortRef.current?.abort();
    setPhase("idle");
  }, []);

  const onDownload = useCallback(() => {
    const blob = new Blob([script], { type: "text/typescript" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "test.spec.ts";
    a.click();
    URL.revokeObjectURL(a.href);
  }, [script]);

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <header
        style={{
          height: 64,
          borderBottom: "1px solid var(--divider)",
          display: "flex",
          alignItems: "center",
          padding: "0 32px",
          gap: 16,
        }}
      >
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontWeight: 700,
            fontSize: 22,
            color: "var(--text)",
            letterSpacing: "-0.02em",
          }}
        >
          qa2pw
        </span>
        <span
          style={{
            fontSize: 14,
            color: "var(--text-muted)",
            borderLeft: "1px solid var(--divider)",
            paddingLeft: 16,
          }}
        >
          Plain English in. Real Playwright out.
        </span>
        <span
          aria-live="polite"
          className="sr-only"
        >
          {phase === "streaming" && step !== null
            ? `Step ${stepCount} — ${step.verb}`
            : phase === "complete"
            ? "Run complete"
            : phase === "partial"
            ? `Stopped: ${stopReason}`
            : phase === "error"
            ? `Error: ${errorMessage}`
            : ""}
        </span>
      </header>

      <main
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1.4fr 1.2fr",
          flex: 1,
          minHeight: 0,
        }}
      >
        <section style={paneStyle} aria-label="Form">
          <FieldGroup label="Test case">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={phase === "streaming"}
              style={{ minHeight: 140, resize: "vertical" }}
            />
          </FieldGroup>
          <FieldGroup label="URL">
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr auto",
                gap: 8,
                alignItems: "center",
              }}
            >
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                disabled={phase === "streaming"}
              />
              {isAllowlisted && <AllowlistBadge />}
            </div>
          </FieldGroup>
          {!isAllowlisted && (
            <label
              style={{
                display: "flex",
                gap: 8,
                fontSize: 13,
                color: "var(--text-muted)",
                lineHeight: 1.5,
                background: "var(--accent-soft)",
                padding: 12,
                borderRadius: 6,
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={attested}
                onChange={(e) => setAttested(e.target.checked)}
                disabled={phase === "streaming"}
              />
              <span>
                I own or operate this site, or have explicit permission to test
                it. (Required for sites outside the curated allowlist.)
              </span>
            </label>
          )}
          {phase === "streaming" ? (
            <button className="btn btn-primary" onClick={onCancel}>
              Generating… (step {stepCount} of 25) — Cancel
            </button>
          ) : (
            <button
              className="btn btn-primary"
              onClick={onGenerate}
              disabled={prompt.trim().length === 0 || url.length === 0}
            >
              {phase === "complete" || phase === "partial" || phase === "error"
                ? "Generate again"
                : "Generate"}
            </button>
          )}
          <p
            style={{
              fontSize: 13,
              color: "var(--text-muted)",
              lineHeight: 1.5,
              margin: 0,
            }}
          >
            Runs locally. Step counter caps at 25, wall clock at 20 min. The
            script is yours to keep.
          </p>
        </section>

        <section style={paneStyle} aria-label="Screencast">
          <div
            style={{
              fontSize: phase === "streaming" || phase === "complete" || phase === "partial" ? 24 : 18,
              fontWeight: 500,
              lineHeight: 1.3,
              color:
                phase === "partial"
                  ? "var(--accent-text)"
                  : phase === "complete"
                  ? "var(--success)"
                  : phase === "error"
                  ? "var(--error)"
                  : phase === "streaming"
                  ? "var(--text)"
                  : "var(--text-faint)",
              minHeight: 64,
            }}
          >
            {phase === "idle" && "Press Generate to start."}
            {phase === "streaming" && step !== null && (
              <>
                <span style={{ color: "var(--accent-text)" }}>{step.verb}</span>
                <span
                  style={{
                    color: "var(--text-muted)",
                    fontSize: 16,
                    fontWeight: 400,
                    display: "block",
                    marginTop: 4,
                  }}
                >
                  {step.reason}
                </span>
              </>
            )}
            {phase === "complete" && (
              <>
                Run complete.
                <span style={muteSub}>
                  {stepCount} steps. Click Download to save the script.
                </span>
              </>
            )}
            {phase === "partial" && (
              <>
                Stopped at step {stepCount} of 25.
                <span style={muteSub}>{stopReason}</span>
              </>
            )}
            {phase === "error" && (
              <>
                Error.
                <span style={muteSub}>{errorMessage}</span>
              </>
            )}
          </div>
          <div
            style={{
              flex: 1,
              border: "1px solid var(--divider)",
              borderRadius: 4,
              background: "var(--surface)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              minHeight: 360,
              marginTop: 16,
              color: "var(--text-faint)",
              fontSize: 14,
              padding: frame.length > 0 ? 0 : 24,
              textAlign: "center",
              overflow: "hidden",
            }}
          >
            {frame.length > 0 ? (
              <img
                src={`data:image/jpeg;base64,${frame}`}
                alt="live browser frame"
                style={{ maxWidth: "100%", maxHeight: "100%", display: "block" }}
              />
            ) : phase === "idle" ? (
              "Click Generate to watch Claude work."
            ) : phase === "streaming" ? (
              "Waiting for the first frame…"
            ) : phase === "complete" || phase === "partial" ? (
              "Run finished. Check the right pane for the script."
            ) : (
              "No frames captured."
            )}
          </div>
        </section>

        <section style={paneStyle} aria-label="Generated script">
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              height: 40,
            }}
          >
            <span
              style={{
                fontSize: 13,
                fontWeight: 500,
                color: "var(--text-muted)",
              }}
            >
              Generated test
            </span>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 13,
                color: "var(--text-faint)",
              }}
            >
              test.spec.ts
            </span>
          </div>

          {phase === "partial" && (
            <div
              style={{
                background: "var(--accent-soft)",
                borderLeft: "3px solid var(--accent-text)",
                borderRadius: 6,
                padding: "12px 16px",
                margin: "8px 0 12px 0",
              }}
            >
              <div
                style={{
                  fontWeight: 500,
                  color: "var(--accent-text)",
                  fontSize: 14,
                }}
              >
                Stopped at step {stepCount} of 25
              </div>
              <div style={{ fontSize: 13, lineHeight: 1.5, marginTop: 4 }}>
                {stopReason === "step_budget"
                  ? "Step budget exhausted. The script below covers what we got — it may not pass yet."
                  : stopReason === "wall_clock"
                  ? "Wall clock fired. The script below covers what we got — it may not pass yet."
                  : `Stopped: ${stopReason}`}
              </div>
            </div>
          )}

          <pre
            style={{
              flex: 1,
              background: "var(--surface-2)",
              borderRadius: 4,
              padding: "16px 20px",
              overflow: "auto",
              fontFamily: "var(--font-mono)",
              fontSize: 14,
              lineHeight: 1.6,
              margin: 0,
              whiteSpace: "pre",
            }}
          >
            {script.length > 0
              ? script
              : "// Your generated Playwright test appears here, line by line."}
          </pre>

          <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
            <button
              className="btn btn-outline"
              onClick={onDownload}
              disabled={script.length === 0}
            >
              {phase === "partial" ? "Download partial" : "Download .spec.ts"}
            </button>
            <button
              className="btn btn-ghost"
              onClick={() => navigator.clipboard.writeText(script)}
              disabled={script.length === 0}
            >
              Copy
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}

function FieldGroup({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <label
        style={{
          fontSize: 13,
          fontWeight: 500,
          color: "var(--text-muted)",
        }}
      >
        {label}
      </label>
      {children}
    </div>
  );
}

function AllowlistBadge() {
  return (
    <span
      style={{
        fontFamily: "var(--font-mono)",
        fontSize: 12,
        color: "var(--success)",
        padding: "4px 8px",
        background: "#DCFCE7",
        borderRadius: 4,
      }}
    >
      allowlisted
    </span>
  );
}

const paneStyle: React.CSSProperties = {
  padding: 24,
  borderRight: "1px solid var(--divider)",
  display: "flex",
  flexDirection: "column",
  gap: 16,
  overflowY: "auto",
  minWidth: 0,
};

const muteSub: React.CSSProperties = {
  color: "var(--text-muted)",
  fontSize: 16,
  fontWeight: 400,
  display: "block",
  marginTop: 4,
};

function parseSse(block: string): StreamEvent | null {
  const lines = block.split("\n");
  let eventType: string | null = null;
  let data = "";
  for (const line of lines) {
    if (line.startsWith("event: ")) eventType = line.slice(7).trim();
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  if (eventType === null || data.length === 0) return null;
  try {
    const parsed = JSON.parse(data);
    return { type: eventType as StreamEvent["type"], ...parsed } as StreamEvent;
  } catch {
    return null;
  }
}

function handleEvent(
  evt: StreamEvent,
  setters: {
    setStep: (s: { verb: string; reason: string } | null) => void;
    setStepCount: (n: number) => void;
    setScript: (s: string) => void;
    setStopReason: (s: string) => void;
    setPhase: (p: Phase) => void;
    setErrorMessage: (s: string) => void;
    setFrame: (b64: string) => void;
  },
): void {
  switch (evt.type) {
    case "status":
      if (evt.status === "complete") setters.setPhase("complete");
      else if (evt.status === "partial") {
        setters.setPhase("partial");
        if (evt.detail !== undefined) setters.setStopReason(evt.detail);
      } else if (evt.status === "error") {
        setters.setPhase("error");
        if (evt.detail !== undefined && evt.detail !== "model_error") {
          setters.setErrorMessage(evt.detail);
        }
      }
      break;
    case "step":
      setters.setStep({ verb: evt.verb, reason: evt.reason });
      setters.setStepCount(evt.step);
      break;
    case "result":
      setters.setScript(evt.script);
      setters.setStopReason(evt.stopReason);
      break;
    case "fatal":
      setters.setPhase("error");
      setters.setErrorMessage(evt.message ?? "unknown error");
      break;
    case "frame":
      setters.setFrame(evt.data);
      break;
  }
}

function safeHost(raw: string): string {
  try {
    return new URL(raw).host;
  } catch {
    return "";
  }
}
