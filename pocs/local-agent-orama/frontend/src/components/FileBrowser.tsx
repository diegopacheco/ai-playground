import { FileEntry } from '../api/client'

interface FileBrowserProps {
  files: FileEntry[]
  selectedFile: string | null
  onSelectFile: (path: string) => void
}

function FileBrowser({ files, selectedFile, onSelectFile }: FileBrowserProps) {
  const sortedFiles = [...files].sort((a, b) => {
    if (a.is_dir && !b.is_dir) return -1
    if (!a.is_dir && b.is_dir) return 1
    return a.name.localeCompare(b.name)
  })

  return (
    <div className="h-full overflow-auto bg-slate-900 border-r border-slate-700">
      <div className="p-2">
        <h4 className="text-sm font-semibold text-slate-400 mb-2 px-2">Files</h4>
        {sortedFiles.length === 0 ? (
          <p className="text-slate-500 text-sm px-2">No files yet</p>
        ) : (
          <ul className="space-y-1">
            {sortedFiles.map((file) => (
              <li key={file.path}>
                <button
                  onClick={() => !file.is_dir && onSelectFile(file.path)}
                  disabled={file.is_dir}
                  className={`w-full text-left px-2 py-1 rounded text-sm flex items-center gap-2 ${
                    selectedFile === file.path
                      ? 'bg-blue-600 text-white'
                      : file.is_dir
                      ? 'text-slate-500 cursor-default'
                      : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  <span>{file.is_dir ? '[D]' : '[F]'}</span>
                  <span className="truncate">{file.name}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

export default FileBrowser
