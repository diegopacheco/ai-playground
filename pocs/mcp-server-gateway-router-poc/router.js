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

export function validName(name) {
  return typeof name === "string" && /^[A-Za-z0-9-]+(_[A-Za-z0-9-]+)*$/.test(name) && !name.includes(SEPARATOR);
}

export function createRouter(upstreams, call = rpc) {
  const byName = new Map(upstreams.map((upstream) => [upstream.name, upstream]));

  function listUpstream(upstream) {
    if (upstream.local) return upstream.local.listTools();
    return call(upstream.url, "tools/list", {}).then(({ tools }) => tools);
  }

  function callUpstream(upstream, tool, args) {
    if (upstream.local) return upstream.local.callTool(tool, args);
    return call(upstream.url, "tools/call", { name: tool, arguments: args });
  }

  async function listTools() {
    const all = [...byName.values()];
    const lists = await Promise.allSettled(
      all.map(async (upstream) => {
        const tools = await listUpstream(upstream);
        return tools.map((tool) => ({
          ...tool,
          name: `${upstream.name}${SEPARATOR}${tool.name}`,
          description: `[${upstream.name}] ${tool.description ?? ""}`.trim(),
        }));
      }),
    );
    return lists.filter((list) => list.status === "fulfilled").flatMap((list) => list.value);
  }

  async function callTool(qualified, args) {
    const parts = splitToolName(qualified);
    const upstream = parts && byName.get(parts.upstream);
    if (!upstream) throw new McpError(ERRORS.INVALID_PARAMS, `no upstream for tool: ${qualified}`);
    return callUpstream(upstream, parts.tool, args);
  }

  function add(upstream) {
    if (!validName(upstream.name)) throw new McpError(ERRORS.INVALID_PARAMS, `invalid upstream name: ${upstream.name}`);
    byName.set(upstream.name, upstream);
  }

  function remove(name) {
    return byName.delete(name);
  }

  function servers() {
    return [...byName.values()].map(({ name, url, kind, source }) => ({ name, kind: kind ?? "url", url, source }));
  }

  return { listTools, callTool, add, remove, get: (name) => byName.get(name), servers };
}
