function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")
}

function slugify(value) {
  return value
    .toLowerCase()
    .replace(/<[^>]+>/g, "")
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
}

function inlineMarkdown(value) {
  let text = escapeHtml(value)
  const code = []

  text = text.replace(/`([^`]+)`/g, (_, content) => {
    code.push(`<code>${content}</code>`)
    return `\u0000CODE${code.length - 1}\u0000`
  })
  text = text.replace(/!\[([^\]]*)\]\((https?:\/\/[^\s)]+)(?:\s+&quot;([^&]*)&quot;)?\)/g, '<img src="$2" alt="$1" loading="lazy">')
  text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)(?:\s+&quot;([^&]*)&quot;)?\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
  text = text.replace(/(^|[\s(])&lt;(https?:\/\/[^&]+)&gt;/g, '$1<a href="$2" target="_blank" rel="noreferrer">$2</a>')
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
  text = text.replace(/__([^_]+)__/g, "<strong>$1</strong>")
  text = text.replace(/~~([^~]+)~~/g, "<del>$1</del>")
  text = text.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>")
  text = text.replace(/(^|[^_])_([^_]+)_/g, "$1<em>$2</em>")
  text = text.replace(/\u0000CODE(\d+)\u0000/g, (_, index) => code[Number(index)])
  return text
}

function splitTableRow(line) {
  return line
    .trim()
    .replace(/^\||\|$/g, "")
    .split(/(?<!\\)\|/)
    .map(cell => cell.trim().replaceAll("\\|", "|"))
}

function isTableDivider(line) {
  const cells = splitTableRow(line)
  return cells.length > 0 && cells.every(cell => /^:?-{3,}:?$/.test(cell))
}

function markdownToHtml(markdown) {
  const lines = String(markdown).replace(/\r\n?/g, "\n").split("\n")
  const output = []
  let paragraph = []
  let listType = ""
  let inCode = false
  let codeLanguage = ""
  let codeLines = []

  const closeParagraph = () => {
    if (!paragraph.length) return
    output.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`)
    paragraph = []
  }

  const closeList = () => {
    if (!listType) return
    output.push(`</${listType}>`)
    listType = ""
  }

  const closeBlocks = () => {
    closeParagraph()
    closeList()
  }

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    const fence = line.match(/^\s*```\s*([^\s`]*)\s*$/)

    if (fence) {
      if (inCode) {
        output.push(`<pre><code${codeLanguage ? ` data-language="${escapeHtml(codeLanguage)}"` : ""}>${escapeHtml(codeLines.join("\n"))}</code></pre>`)
        inCode = false
        codeLanguage = ""
        codeLines = []
      } else {
        closeBlocks()
        inCode = true
        codeLanguage = fence[1]
      }
      continue
    }

    if (inCode) {
      codeLines.push(line)
      continue
    }

    if (line.includes("|") && index + 1 < lines.length && isTableDivider(lines[index + 1])) {
      closeBlocks()
      const headers = splitTableRow(line)
      const alignments = splitTableRow(lines[index + 1]).map(cell => {
        if (cell.startsWith(":") && cell.endsWith(":")) return "center"
        if (cell.endsWith(":")) return "right"
        return "left"
      })
      const rows = []
      index += 2

      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(splitTableRow(lines[index]))
        index += 1
      }

      output.push("<div class=\"table-scroll\"><table><thead><tr>")
      headers.forEach((cell, cellIndex) => output.push(`<th style="text-align:${alignments[cellIndex] || "left"}">${inlineMarkdown(cell)}</th>`))
      output.push("</tr></thead><tbody>")
      rows.forEach(row => {
        output.push("<tr>")
        headers.forEach((_, cellIndex) => output.push(`<td style="text-align:${alignments[cellIndex] || "left"}">${inlineMarkdown(row[cellIndex] || "")}</td>`))
        output.push("</tr>")
      })
      output.push("</tbody></table></div>")
      index -= 1
      continue
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/)
    if (heading) {
      closeBlocks()
      const level = heading[1].length
      output.push(`<h${level} id="${slugify(heading[2])}">${inlineMarkdown(heading[2])}</h${level}>`)
      continue
    }

    if (/^\s*(?:---+|___+|\*\*\*+)\s*$/.test(line)) {
      closeBlocks()
      output.push("<hr>")
      continue
    }

    const quote = line.match(/^>\s?(.*)$/)
    if (quote) {
      closeBlocks()
      const quoteLines = [quote[1]]
      while (index + 1 < lines.length && /^>\s?/.test(lines[index + 1])) {
        index += 1
        quoteLines.push(lines[index].replace(/^>\s?/, ""))
      }
      output.push(`<blockquote>${markdownToHtml(quoteLines.join("\n"))}</blockquote>`)
      continue
    }

    const unordered = line.match(/^\s*[-+*]\s+(.+)$/)
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/)
    if (unordered || ordered) {
      closeParagraph()
      const nextType = ordered ? "ol" : "ul"
      if (listType && listType !== nextType) closeList()
      if (!listType) {
        listType = nextType
        output.push(`<${listType}>`)
      }
      const item = (unordered || ordered)[1]
      const task = item.match(/^\[([ xX])\]\s+(.+)$/)
      output.push(task ? `<li class="task"><input type="checkbox" disabled${task[1].toLowerCase() === "x" ? " checked" : ""}>${inlineMarkdown(task[2])}</li>` : `<li>${inlineMarkdown(item)}</li>`)
      continue
    }

    if (!line.trim()) {
      closeBlocks()
      continue
    }

    if (/^ {4}/.test(line)) {
      closeBlocks()
      const indented = [line.slice(4)]
      while (index + 1 < lines.length && /^ {4}/.test(lines[index + 1])) {
        index += 1
        indented.push(lines[index].slice(4))
      }
      output.push(`<pre><code>${escapeHtml(indented.join("\n"))}</code></pre>`)
      continue
    }

    closeList()
    paragraph.push(line.trim())
  }

  if (inCode) output.push(`<pre><code${codeLanguage ? ` data-language="${escapeHtml(codeLanguage)}"` : ""}>${escapeHtml(codeLines.join("\n"))}</code></pre>`)
  closeBlocks()
  return output.join("\n")
}

const DocumentRenderer = { escapeHtml, inlineMarkdown, markdownToHtml }

if (typeof globalThis !== "undefined") globalThis.DocumentRenderer = DocumentRenderer
if (typeof module !== "undefined") module.exports = DocumentRenderer
