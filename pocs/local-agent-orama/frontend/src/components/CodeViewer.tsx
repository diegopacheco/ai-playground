import { useMemo } from 'react'

interface CodeViewerProps {
  content: string | null
  language: string
  filePath: string | null
}

interface Token {
  text: string
  type: 'keyword' | 'string' | 'comment' | 'number' | 'function' | 'type' | 'plain'
}

const KEYWORDS: Record<string, Set<string>> = {
  rust: new Set(['fn', 'let', 'mut', 'const', 'struct', 'enum', 'impl', 'trait', 'pub', 'use', 'mod', 'crate', 'self', 'super', 'where', 'async', 'await', 'move', 'ref', 'static', 'type', 'unsafe', 'extern', 'dyn', 'if', 'else', 'match', 'loop', 'while', 'for', 'in', 'break', 'continue', 'return', 'true', 'false', 'Some', 'None', 'Ok', 'Err', 'Self']),
  javascript: new Set(['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'try', 'catch', 'finally', 'throw', 'new', 'delete', 'typeof', 'instanceof', 'void', 'this', 'class', 'extends', 'import', 'export', 'default', 'from', 'as', 'async', 'await', 'yield', 'true', 'false', 'null', 'undefined']),
  typescript: new Set(['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'try', 'catch', 'finally', 'throw', 'new', 'delete', 'typeof', 'instanceof', 'void', 'this', 'class', 'extends', 'import', 'export', 'default', 'from', 'as', 'async', 'await', 'yield', 'true', 'false', 'null', 'undefined', 'interface', 'type', 'enum', 'implements', 'private', 'public', 'protected', 'readonly', 'abstract', 'namespace']),
  python: new Set(['def', 'class', 'return', 'if', 'elif', 'else', 'for', 'while', 'break', 'continue', 'pass', 'try', 'except', 'finally', 'raise', 'import', 'from', 'as', 'with', 'lambda', 'yield', 'global', 'nonlocal', 'assert', 'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is', 'async', 'await', 'self']),
  go: new Set(['func', 'var', 'const', 'type', 'struct', 'interface', 'map', 'chan', 'package', 'import', 'return', 'if', 'else', 'for', 'range', 'switch', 'case', 'default', 'break', 'continue', 'go', 'select', 'defer', 'make', 'new', 'true', 'false', 'nil', 'iota']),
  java: new Set(['public', 'private', 'protected', 'class', 'interface', 'extends', 'implements', 'static', 'final', 'void', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'try', 'catch', 'finally', 'throw', 'throws', 'new', 'this', 'super', 'import', 'package', 'true', 'false', 'null', 'abstract', 'synchronized', 'volatile', 'transient', 'native', 'enum', 'instanceof']),
}

function tokenizeLine(line: string, language: string): Token[] {
  const tokens: Token[] = []
  const keywords = KEYWORDS[language] || KEYWORDS['javascript'] || new Set()
  let i = 0
  while (i < line.length) {
    if (line[i] === '/' && line[i + 1] === '/') {
      tokens.push({ text: line.slice(i), type: 'comment' })
      break
    }
    if (line[i] === '#' && (language === 'python' || language === 'bash')) {
      tokens.push({ text: line.slice(i), type: 'comment' })
      break
    }
    if (line[i] === '"' || line[i] === "'" || line[i] === '`') {
      const quote = line[i]
      let j = i + 1
      while (j < line.length && (line[j] !== quote || line[j - 1] === '\\')) {
        j++
      }
      tokens.push({ text: line.slice(i, j + 1), type: 'string' })
      i = j + 1
      continue
    }
    if (/[0-9]/.test(line[i])) {
      let j = i
      while (j < line.length && /[0-9.]/.test(line[j])) {
        j++
      }
      tokens.push({ text: line.slice(i, j), type: 'number' })
      i = j
      continue
    }
    if (/[a-zA-Z_]/.test(line[i])) {
      let j = i
      while (j < line.length && /[a-zA-Z0-9_]/.test(line[j])) {
        j++
      }
      const word = line.slice(i, j)
      const nextChar = line[j]
      if (keywords.has(word)) {
        tokens.push({ text: word, type: 'keyword' })
      } else if (nextChar === '(') {
        tokens.push({ text: word, type: 'function' })
      } else if (/^[A-Z]/.test(word)) {
        tokens.push({ text: word, type: 'type' })
      } else {
        tokens.push({ text: word, type: 'plain' })
      }
      i = j
      continue
    }
    tokens.push({ text: line[i], type: 'plain' })
    i++
  }
  return tokens
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function renderToken(token: Token): string {
  const escaped = escapeHtml(token.text)
  switch (token.type) {
    case 'keyword':
      return `<span style="color:#c084fc">${escaped}</span>`
    case 'string':
      return `<span style="color:#4ade80">${escaped}</span>`
    case 'comment':
      return `<span style="color:#64748b">${escaped}</span>`
    case 'number':
      return `<span style="color:#fb923c">${escaped}</span>`
    case 'function':
      return `<span style="color:#60a5fa">${escaped}</span>`
    case 'type':
      return `<span style="color:#facc15">${escaped}</span>`
    default:
      return escaped
  }
}

function highlightLine(line: string, language: string): string {
  const tokens = tokenizeLine(line, language)
  return tokens.map(renderToken).join('')
}

function CodeViewer({ content, language, filePath }: CodeViewerProps) {
  const highlightedLines = useMemo(() => {
    if (!content) return []
    return content.split('\n').map(line => highlightLine(line, language))
  }, [content, language])

  if (!filePath) {
    return (
      <div className="h-full flex items-center justify-center text-slate-500">
        Select a file to view its contents
      </div>
    )
  }

  if (content === null) {
    return (
      <div className="h-full flex items-center justify-center text-slate-500">
        Loading...
      </div>
    )
  }

  const lineCount = highlightedLines.length
  const lineNumberWidth = Math.max(3, String(lineCount).length)

  return (
    <div className="h-full flex flex-col">
      <div className="px-4 py-2 bg-slate-800 border-b border-slate-700 text-sm text-slate-400">
        {filePath}
      </div>
      <div className="flex-1 overflow-auto bg-slate-950">
        <table className="w-full text-sm font-mono">
          <tbody>
            {highlightedLines.map((line, index) => (
              <tr key={index} className="hover:bg-slate-900">
                <td
                  className="text-right pr-4 pl-2 text-slate-600 select-none border-r border-slate-800 sticky left-0 bg-slate-950"
                  style={{ width: `${lineNumberWidth + 2}ch` }}
                >
                  {index + 1}
                </td>
                <td className="pl-4 pr-4 whitespace-pre text-slate-300">
                  <span dangerouslySetInnerHTML={{ __html: line || '&nbsp;' }} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default CodeViewer
