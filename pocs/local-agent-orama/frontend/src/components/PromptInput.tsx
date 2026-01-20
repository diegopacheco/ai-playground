import { useState } from 'react'

interface PromptInputProps {
  onSubmit: (prompt: string, projectName: string) => void
  disabled: boolean
}

function PromptInput({ onSubmit, disabled }: PromptInputProps) {
  const [prompt, setPrompt] = useState('')
  const [projectName, setProjectName] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (prompt.trim() && projectName.trim()) {
      onSubmit(prompt.trim(), projectName.trim())
    }
  }

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto">
      <div className="flex flex-col gap-4">
        <input
          type="text"
          value={projectName}
          onChange={(e) => setProjectName(e.target.value)}
          placeholder="Project name..."
          disabled={disabled}
          className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-blue-500 disabled:opacity-50"
        />
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your prompt for all agents..."
          disabled={disabled}
          rows={4}
          className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-blue-500 resize-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={disabled || !prompt.trim() || !projectName.trim()}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:cursor-not-allowed rounded-lg font-semibold transition-colors"
        >
          {disabled ? 'Running...' : 'Run All Agents'}
        </button>
      </div>
    </form>
  )
}

export default PromptInput
