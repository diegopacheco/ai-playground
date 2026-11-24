import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { loadConfig } from "./config.ts";
import { createLogger } from "./logger.ts";
import { buildServer, loadParts } from "./server.ts";

const config = loadConfig();
const logger = createLogger(config.serviceName);

try {
  const parts = await loadParts(config, logger);
  serveStdio(() => buildServer(config, parts), { onerror: (err) => logger.error("stdio error", { error: err.message }) });
  logger.info("server ready", { transport: "stdio" });
} catch (err) {
  logger.error("server failed to start", { error: err instanceof Error ? err.message : String(err) });
  process.exit(1);
}
