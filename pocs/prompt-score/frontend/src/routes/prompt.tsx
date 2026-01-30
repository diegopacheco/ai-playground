import { useState } from 'react'
import PromptInput from '../components/PromptInput'

interface PromptPageProps {
  onAnalyze: (prompt: string) => void
  loading: boolean
}

function PromptPage({ onAnalyze, loading }: PromptPageProps) {
  const [prompt, setPrompt] = useState('')

  const charCount = prompt.length
  const wordCount = prompt.trim() ? prompt.trim().split(/\s+/).length : 0

  const handleSubmit = () => {
    if (prompt.trim()) {
      onAnalyze(prompt)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Enter Your Prompt</h2>
        <PromptInput value={prompt} onChange={setPrompt} />
        <div className="flex justify-between items-center mt-4">
          <div className="text-gray-400 text-sm">
            <span className="mr-4">{charCount} characters</span>
            <span>{wordCount} words</span>
          </div>
          <button
            onClick={handleSubmit}
            disabled={loading || !prompt.trim()}
            className={`px-6 py-2 rounded-lg font-medium transition-colors ${
              loading || !prompt.trim()
                ? 'bg-gray-600 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default PromptPage
