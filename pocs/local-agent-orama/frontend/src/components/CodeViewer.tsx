import { useMemo } from 'react'

interface CodeViewerProps {
  content: string | null
  language: string
  filePath: string | null
}

const KEYWORDS: Record<string, string[]> = {
  rust: ['fn', 'let', 'mut', 'const', 'struct', 'enum', 'impl', 'trait', 'pub', 'use', 'mod', 'crate', 'self', 'super', 'where', 'async', 'await', 'move', 'ref', 'static', 'type', 'unsafe', 'extern', 'dyn', 'if', 'else', 'match', 'loop', 'while', 'for', 'in', 'break', 'continue', 'return', 'true', 'false', 'Some', 'None', 'Ok', 'Err', 'Self'],
  javascript: ['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'try', 'catch', 'finally', 'throw', 'new', 'delete', 'typeof', 'instanceof', 'void', 'this', 'class', 'extends', 'import', 'export', 'default', 'from', 'as', 'async', 'await', 'yield', 'true', 'false', 'null', 'undefined'],
  typescript: ['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'try', 'catch', 'finally', 'throw', 'new', 'delete', 'typeof', 'instanceof', 'void', 'this', 'class', 'extends', 'import', 'export', 'default', 'from', 'as', 'async', 'await', 'yield', 'true', 'false', 'null', 'undefined', 'interface', 'type', 'enum', 'implements', 'private', 'public', 'protected', 'readonly', 'abstract', 'namespace'],
  python: ['def', 'class', 'return', 'if', 'elif', 'else', 'for', 'while', 'break', 'continue', 'pass', 'try', 'except', 'finally', 'raise', 'import', 'from', 'as', 'with', 'lambda', 'yield', 'global', 'nonlocal', 'assert', 'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is', 'async', 'await', 'self'],
  go: ['func', 'var', 'const', 'type', 'struct', 'interface', 'map', 'chan', 'package', 'import', 'return', 'if', 'else', 'for', 'range', 'switch', 'case', 'default', 'break', 'continue', 'go', 'select', 'defer', 'make', 'new', 'true', 'false', 'nil', 'iota'],
  java: ['public', 'private', 'protected', 'class', 'interface', 'extends', 'implements', 'static', 'final', 'void', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'try', 'catch', 'finally', 'throw', 'throws', 'new', 'this', 'super', 'import', 'package', 'true', 'false', 'null', 'abstract', 'synchronized', 'volatile', 'transient', 'native', 'enum', 'instanceof'],
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function highlightLine(line: string, language: string): string {
  let escaped = escapeHtml(line)
  escaped = escaped.replace(/(\/\/.*$|#.*$)/gm, '<span class="text-slate-500">$1</span>')
  escaped = escaped.replace(/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)/g, '<span class="text-green-400">$1</span>')
  escaped = escaped.replace(/\b(\d+\.?\d*)\b/g, '<span class="text-orange-400">$1</span>')
  const keywords = KEYWORDS[language] || KEYWORDS['javascript'] || []
  keywords.forEach(kw => {
    const regex = new RegExp(`\\b(${kw})\\b`, 'g')
    escaped = escaped.replace(regex, '<span class="text-purple-400">$1</span>')
  })
  escaped = escaped.replace(/\b([A-Z][a-zA-Z0-9_]*)\b/g, '<span class="text-yellow-400">$1</span>')
  escaped = escaped.replace(/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g, '<span class="text-blue-400">$1</span>(')
  return escaped
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
