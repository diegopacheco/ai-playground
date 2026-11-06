import { spawn } from "node:child_process";

const ARG_LINE = /^-\s+([A-Za-z0-9_]+)\s*\(\s*(string|number|boolean|integer)\s*(,\s*required)?\s*\)\s*:?\s*(.*)$/;

function parseTool(name, lines) {
  const body = lines.map((line) => line.trim());
  const description = body.find((line) => line !== "") ?? "";
  const properties = {};
  const required = [];
  const instructions = [];
  body.slice(body.indexOf(description) + 1).forEach((line) => {
    const arg = line.match(ARG_LINE);
    if (!arg) return instructions.push(line);
    properties[arg[1]] = { type: arg[2], description: arg[4] };
    if (arg[3]) required.push(arg[1]);
  });
  return {
    name,
    description,
    inputSchema: { type: "object", properties, required },
    instructions: instructions.join("\n").trim() || description,
  };
}

export function parseMarkdownServer(markdown) {
  const sections = [];
  let current = null;
  markdown.split("\n").forEach((line) => {
    const heading = line.match(/^##\s+([A-Za-z0-9_-]+)\s*$/);
    if (heading) {
      current = { name: heading[1], lines: [] };
      sections.push(current);
    } else if (current) {
      current.lines.push(line);
    }
  });
  if (sections.length === 0) throw new Error("markdown server needs at least one '## tool_name' section");
  return sections.map(({ name, lines }) => parseTool(name, lines));
}

function checkArgs(tool, args) {
  tool.inputSchema.required.forEach((key) => {
    if (args[key] === undefined) throw new Error(`missing required argument: ${key}`);
  });
  Object.entries(args).forEach(([key, value]) => {
    const expected = tool.inputSchema.properties[key]?.type;
    if (!expected) throw new Error(`unknown argument: ${key}`);
    const actual = expected === "integer" ? (Number.isInteger(value) ? "integer" : typeof value) : typeof value;
    if (actual !== expected) throw new Error(`${key} must be a ${expected}`);
  });
}

export function toolPrompt(tool, args) {
  return [
    `You are the MCP tool "${tool.name}". ${tool.description}`,
    "Do what the instructions say with the arguments. Answer with the tool result only, no preamble.",
    "Instructions:",
    tool.instructions,
    "Arguments (JSON):",
    JSON.stringify(args),
  ].join("\n\n");
}

export function runClaude(prompt, { bin = process.env.CLAUDE_BIN || "claude", timeoutMs = Number(process.env.MD_TOOL_TIMEOUT_MS) || 120000 } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(bin, ["-p", "--strict-mcp-config"], { stdio: ["pipe", "pipe", "pipe"] });
    let out = "";
    let err = "";
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`claude -p timed out after ${timeoutMs} ms`));
    }, timeoutMs);
    child.stdout.on("data", (chunk) => (out += chunk));
    child.stderr.on("data", (chunk) => (err += chunk));
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(new Error(`cannot run ${bin}: ${error.message}`));
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) return reject(new Error(`claude -p exited ${code}: ${err.trim() || out.trim()}`));
      resolve(out.trim());
    });
    child.stdin.end(prompt);
  });
}

export function markdownTools(markdown, run = runClaude) {
  return parseMarkdownServer(markdown).map((tool) => ({
    name: tool.name,
    description: tool.description,
    inputSchema: tool.inputSchema,
    handler: (args) => {
      checkArgs(tool, args);
      return run(toolPrompt(tool, args));
    },
  }));
}
