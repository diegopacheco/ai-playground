import { catalog } from "./catalog.ts";
import { config } from "./config.ts";
import { CORRELATION_HEADER, createLogger } from "./logger.ts";
import { fileReader, oxlintLinter } from "./oxlint.ts";
import { buildReport, readSources } from "./report.ts";

const logger = createLogger(config.serviceName);
const linter = oxlintLinter(config.rootDir, config.oxlintConfig, config.slopDir);
const reader = fileReader(config.rootDir);

async function report(): Promise<Response> {
  const [findings, sources] = await Promise.all([linter.run(), readSources(catalog, reader)]);

  return Response.json(buildReport(catalog, findings, sources));
}

Bun.serve({
  port: config.port,
  async fetch(request) {
    const path = new URL(request.url).pathname;
    const correlationId = request.headers.get(CORRELATION_HEADER) ?? crypto.randomUUID();
    const context = { correlationId, path, userId: undefined };

    try {
      const response = path === "/api/report" ? await report() : Response.json({ error: "not found" }, { status: 404 });
      logger.info("request", context, { status: response.status });
      response.headers.set(CORRELATION_HEADER, correlationId);

      return response;
    } catch (error) {
      logger.error("request failed", context, { error: String(error) });

      return Response.json({ error: "lint failed" }, { status: 500, headers: { [CORRELATION_HEADER]: correlationId } });
    }
  },
});

logger.info("started", { correlationId: "startup", path: "", userId: undefined }, { port: config.port });
