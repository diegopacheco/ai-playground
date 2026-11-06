import { rpc } from "./lib/mcp-client.js";
import { McpError, ERRORS } from "./lib/mcp-server.js";

export const SEPARATOR = "__";

export function parseUpstreams(spec) {
  if (!spec) throw new Error("UPSTREAMS env var is required, i.e. math=http://localhost:8101/mcp");
  return spec.split(",").map((entry) => {
    const [name, url] = entry.split("=").map((part) => part?.trim());
    if (!name || !url || name.includes(SEPARATOR)) throw new Error(`invalid upstream entry: ${entry}`);
    return { name, url };
  });
}

export function splitToolName(qualified) {
  const index = qualified.indexOf(SEPARATOR);
  if (index <= 0) return null;
  return { upstream: qualified.slice(0, index), tool: qualified.slice(index + SEPARATOR.length) };
}

export function createRouter(upstreams, call = rpc) {
  const byName = new Map(upstreams.map((upstream) => [upstream.name, upstream]));

  async function listTools() {
    const lists = await Promise.allSettled(
      upstreams.map(async ({ name, url }) => {
        const { tools } = await call(url, "tools/list", {});
        return tools.map((tool) => ({
          ...tool,
          name: `${name}${SEPARATOR}${tool.name}`,
          description: `[${name}] ${tool.description ?? ""}`.trim(),
        }));
      }),
    );
    return lists.filter((list) => list.status === "fulfilled").flatMap((list) => list.value);
  }

  async function callTool(qualified, args) {
    const parts = splitToolName(qualified);
    const upstream = parts && byName.get(parts.upstream);
    if (!upstream) throw new McpError(ERRORS.INVALID_PARAMS, `no upstream for tool: ${qualified}`);
    return call(upstream.url, "tools/call", { name: parts.tool, arguments: args });
  }

  return { listTools, callTool };
}
