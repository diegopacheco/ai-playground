import { useState } from 'react'

interface PromptViewProps {
  onSubmit: (prompt: string) => void
}

const suggestedTrainings = [
  'Using AI to write better emails',
  'Using AI with Excel spreadsheets',
  'Using AI to improve your work productivity',
  'Using AI to review PPT presentations',
]

function PromptView({ onSubmit }: PromptViewProps) {
  const [prompt, setPrompt] = useState('')

  const wordCount = prompt.trim() ? prompt.trim().split(/\s+/).length : 0
  const charCount = prompt.length

  const handleSubmit = () => {
    if (prompt.trim()) {
      onSubmit(prompt.trim())
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6">
      <div className="max-w-4xl w-full">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4 tracking-tight">
            Self Training Machine
          </h1>
          <p className="text-xl text-purple-200">
            Create personalized AI-powered training with quizzes
          </p>
        </div>

        <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-white/20">
          <label className="block text-white text-lg font-medium mb-4">
            What would you like to learn today?
          </label>

          <textarea
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            placeholder="Enter your training topic..."
            className="w-full h-40 p-4 rounded-xl bg-white/90 text-gray-800 placeholder-gray-500 border-2 border-transparent focus:border-purple-400 focus:outline-none resize-none text-lg transition-all"
          />

          <div className="flex justify-between items-center mt-3 text-purple-200 text-sm">
            <span>{wordCount} words</span>
            <span>{charCount} characters</span>
          </div>

          <button
            onClick={handleSubmit}
            disabled={!prompt.trim()}
            className="w-full mt-6 py-4 px-6 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-bold text-lg rounded-xl hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-lg"
          >
            Generate Training
          </button>

          <div className="mt-10">
            <p className="text-purple-200 text-center mb-4">Or choose a suggested topic:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {suggestedTrainings.map((topic, index) => (
                <button
                  key={index}
                  onClick={() => onSubmit(topic)}
                  className="p-4 bg-white/5 hover:bg-white/15 border border-white/20 rounded-xl text-white text-left transition-all hover:border-purple-400 hover:shadow-lg"
                >
                  <span className="text-purple-300 mr-2">{index + 1}.</span>
                  {topic}
                </button>
              ))}
            </div>
          </div>
        </div>

        <p className="text-center text-purple-300/60 mt-8 text-sm">
          Powered by Claude Opus 4.5
        </p>
      </div>
    </div>
  )
}

export default PromptView
