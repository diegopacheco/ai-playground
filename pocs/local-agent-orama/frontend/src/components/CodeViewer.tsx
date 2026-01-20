interface CodeViewerProps {
  content: string | null
  language: string
  filePath: string | null
}

function CodeViewer({ content, filePath }: CodeViewerProps) {
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

  return (
    <div className="h-full flex flex-col">
      <div className="px-4 py-2 bg-slate-800 border-b border-slate-700 text-sm text-slate-400">
        {filePath}
      </div>
      <pre className="flex-1 overflow-auto p-4 bg-slate-950 text-sm">
        <code className="text-slate-300 whitespace-pre">{content}</code>
      </pre>
    </div>
  )
}

export default CodeViewer
