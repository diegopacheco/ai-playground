import { NextRequest } from "next/server";
import {
  actionLogToScript,
  checkUrl,
  chromiumLauncher,
  HttpOllamaClient,
  PlaywrightSession,
  runGenerate,
  withDisposable,
  DEFAULTS,
  type RunEvent,
} from "@qa2pw/runner";
import { formatSse, sseHeaders } from "../../lib/sse.ts";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

interface GenerateBody {
  prompt: string;
  url: string;
  attested?: boolean;
}

export async function POST(request: NextRequest): Promise<Response> {
  let body: GenerateBody;
  try {
    body = (await request.json()) as GenerateBody;
  } catch {
    return jsonError(400, "invalid_json", "request body must be JSON");
  }

  if (typeof body.prompt !== "string" || body.prompt.trim().length === 0) {
    return jsonError(400, "missing_prompt", "prompt must be a non-empty string");
  }
  if (typeof body.url !== "string" || body.url.length === 0) {
    return jsonError(400, "missing_url", "url must be a non-empty string");
  }

  const verdict = checkUrl(body.url, body.attested === true);
  if (verdict.ok === false) {
    const status = verdict.reason === "attestation_required" ? 403 : 400;
    return jsonError(status, verdict.reason, urlVerdictMessage(verdict.reason));
  }

  const model = process.env.QA2PW_MODEL ?? DEFAULTS.model;
  const ollamaUrl = process.env.QA2PW_OLLAMA_URL ?? DEFAULTS.ollamaUrl;

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const enqueue = (event: RunEvent) => {
        controller.enqueue(formatSse({ event: event.type, data: event }));
      };

      try {
        await withDisposable(
          () => PlaywrightSession.launch(chromiumLauncher, { headless: true }),
          async (browser) => {
            await browser.startScreencast((frame) => {
              controller.enqueue(
                formatSse({
                  event: "frame",
                  data: {
                    type: "frame",
                    data: frame.data,
                    timestamp: frame.timestamp,
                  },
                }),
              );
            });
            const result = await runGenerate(
              {
                prompt: body.prompt,
                url: body.url,
                attested: body.attested === true,
                onEvent: enqueue,
                model,
                ollamaUrl,
              },
              {
                browser,
                ollama: new HttpOllamaClient(ollamaUrl),
              },
            );
            const script = actionLogToScript(result);
            controller.enqueue(
              formatSse({
                event: "result",
                data: { script, stopReason: result.stopReason, steps: result.log.length },
              }),
            );
          },
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        controller.enqueue(
          formatSse({
            event: "fatal",
            data: { message },
          }),
        );
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, { headers: sseHeaders() });
}

function jsonError(
  status: number,
  code: string,
  message: string,
): Response {
  return new Response(JSON.stringify({ error: code, message }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function urlVerdictMessage(
  reason: "malformed_url" | "blocked_by_safety" | "attestation_required",
): string {
  switch (reason) {
    case "malformed_url":
      return "url is not a valid http(s) address";
    case "blocked_by_safety":
      return "url is on the safety blocklist (banks, .gov) and cannot be driven";
    case "attestation_required":
      return "url is not on the allowlist; set attested=true to confirm you operate or are authorized to test this site";
  }
}
