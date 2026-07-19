const test = require("node:test")
const assert = require("node:assert/strict")
const { escapeHtml, inlineMarkdown, markdownToHtml } = require("../src/renderer.js")

test("escapes active markup", () => {
  assert.equal(escapeHtml('<script src="x">'), "&lt;script src=&quot;x&quot;&gt;")
  assert.equal(inlineMarkdown("<img src=x onerror=alert(1)>"), "&lt;img src=x onerror=alert(1)&gt;")
})

test("renders common inline formatting", () => {
  const output = inlineMarkdown("**Bold**, *italic*, `code`, and [link](https://github.com)")
  assert.match(output, /<strong>Bold<\/strong>/)
  assert.match(output, /<em>italic<\/em>/)
  assert.match(output, /<code>code<\/code>/)
  assert.match(output, /href="https:\/\/github.com"/)
})

test("renders headings, lists, tasks, and code fences", () => {
  const markdown = "# Title\n\n- item\n- [x] done\n\n```js\nconst value = 1 < 2\n```"
  const output = markdownToHtml(markdown)
  assert.match(output, /<h1 id="title">Title<\/h1>/)
  assert.match(output, /<ul>/)
  assert.match(output, /type="checkbox" disabled checked/)
  assert.match(output, /data-language="js"/)
  assert.match(output, /1 &lt; 2/)
})

test("renders tables and quotes", () => {
  const markdown = "| Name | State |\n| :--- | ---: |\n| Reader | On |\n\n> Clear and focused"
  const output = markdownToHtml(markdown)
  assert.match(output, /<table>/)
  assert.match(output, /text-align:right/)
  assert.match(output, /<blockquote>/)
})

test("does not accept unsafe link protocols", () => {
  const output = inlineMarkdown("[run](javascript:alert(1))")
  assert.doesNotMatch(output, /<a/)
})
