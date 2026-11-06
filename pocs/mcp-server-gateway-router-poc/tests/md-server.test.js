import { test } from "node:test";
import assert from "node:assert/strict";
import { markdownTools, parseMarkdownServer } from "../lib/md-server.js";
import { toolRegistry } from "../lib/mcp-server.js";

const markdown = `# writer
Server intro that is not a tool.

## translate
Translates a text into another language.
- text (string, required): the text to translate
- language (string, required): the target language
- formal (boolean): use a formal tone

Translate the text. Answer with the translation only.
`;

test("each ## section becomes a tool whose bullet arguments form its input schema", () => {
  const [tool] = parseMarkdownServer(markdown);
  assert.equal(tool.name, "translate");
  assert.equal(tool.description, "Translates a text into another language.");
  assert.deepEqual(tool.inputSchema.required, ["text", "language"]);
  assert.equal(tool.inputSchema.properties.formal.type, "boolean");
  assert.equal(tool.instructions, "Translate the text. Answer with the translation only.");
});

test("a markdown file without tool sections is rejected instead of registering an empty server", () => {
  assert.throws(() => parseMarkdownServer("# nothing here\njust prose"), /at least one '## tool_name'/);
});

test("bad arguments fail as tool errors before any claude -p run is spent", async () => {
  const prompts = [];
  const registry = toolRegistry(markdownTools(markdown, async (prompt) => prompts.push(prompt)));
  const missing = await registry.callTool("translate", { text: "hi" });
  assert.equal(missing.isError, true);
  assert.match(missing.content[0].text, /missing required argument: language/);
  const wrongType = await registry.callTool("translate", { text: "hi", language: "pt", formal: "yes" });
  assert.match(wrongType.content[0].text, /formal must be a boolean/);
  assert.equal(prompts.length, 0);
});

test("the prompt sent to claude carries the English instructions and the call arguments", async () => {
  const prompts = [];
  const registry = toolRegistry(markdownTools(markdown, async (prompt) => (prompts.push(prompt), "Olá")));
  const result = await registry.callTool("translate", { text: "hello", language: "Portuguese" });
  assert.equal(result.content[0].text, "Olá");
  assert.match(prompts[0], /Translate the text\. Answer with the translation only\./);
  assert.match(prompts[0], /"language":"Portuguese"/);
  assert.doesNotMatch(prompts[0], /- text \(string/);
});
