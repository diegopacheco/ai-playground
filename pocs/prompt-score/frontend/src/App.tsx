import { useState } from 'react'
import { AnalyzeResponse, ProgressEvent, ProgressState, DimensionScores, ModelResult } from './types'
import PromptPage from './routes/prompt'
import ScorePage from './routes/score'

function App() {
  const [activeTab, setActiveTab] = useState<'prompt' | 'score'>('prompt')
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState<ProgressState>({
    currentStep: 0,
    totalSteps: 0,
    message: '',
    isComplete: false
  })

  const handleAnalyze = async (prompt: string) => {
    setLoading(true)
    setActiveTab('score')
    setResult(null)
    setProgress({ currentStep: 0, totalSteps: 6, message: 'Starting analysis...', isComplete: false })

    let scores: DimensionScores | null = null
    const modelResults: ModelResult[] = []

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      })

      if (!response.ok) {
        throw new Error('Analysis failed')
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('No response body')
      }

      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6)
            if (jsonStr.trim()) {
              try {
                const event: ProgressEvent = JSON.parse(jsonStr)
                handleProgressEvent(event, scores, modelResults, setProgress, setResult)
                if (event.type === 'scores') {
                  scores = event.scores
                }
                if (event.type === 'agent_done') {
                  modelResults.push(event.result)
                }
              } catch {
                console.error('Failed to parse SSE event')
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Analysis failed:', error)
      setProgress(p => ({ ...p, message: 'Analysis failed. Please try again.', isComplete: true }))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-6xl mx-auto px-4">
          <h1 className="text-2xl font-bold py-4">Prompt Score</h1>
          <nav className="flex gap-4">
            <button
              onClick={() => setActiveTab('prompt')}
              className={`px-4 py-2 border-b-2 transition-colors ${
                activeTab === 'prompt'
                  ? 'border-blue-500 text-blue-400'
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              Prompt
            </button>
            <button
              onClick={() => setActiveTab('score')}
              className={`px-4 py-2 border-b-2 transition-colors ${
                activeTab === 'score'
                  ? 'border-blue-500 text-blue-400'
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              Score
            </button>
          </nav>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        {activeTab === 'prompt' ? (
          <PromptPage onAnalyze={handleAnalyze} loading={loading} />
        ) : (
          <ScorePage result={result} loading={loading} progress={progress} />
        )}
      </main>
    </div>
  )
}

function handleProgressEvent(
  event: ProgressEvent,
  _scores: DimensionScores | null,
  _modelResults: ModelResult[],
  setProgress: React.Dispatch<React.SetStateAction<ProgressState>>,
  setResult: React.Dispatch<React.SetStateAction<AnalyzeResponse | null>>
) {
  switch (event.type) {
    case 'start':
      setProgress({
        currentStep: 0,
        totalSteps: event.total_steps,
        message: event.message,
        isComplete: false
      })
      break
    case 'scores':
      setProgress(p => ({
        ...p,
        currentStep: event.step,
        message: event.message
      }))
      setResult(prev => prev ? { ...prev, scores: event.scores } : { scores: event.scores, model_results: [] })
      break
    case 'agent_start':
      setProgress(p => ({
        ...p,
        currentStep: event.step,
        message: event.message
      }))
      break
    case 'agent_done':
      setProgress(p => ({
        ...p,
        currentStep: event.step,
        message: event.message
      }))
      setResult(prev => {
        if (!prev) return null
        return {
          ...prev,
          model_results: [...prev.model_results, event.result]
        }
      })
      break
    case 'complete':
      setProgress({
        currentStep: 6,
        totalSteps: 6,
        message: event.message,
        isComplete: true
      })
      setResult({
        scores: event.scores,
        model_results: event.model_results
      })
      break
  }
}

export default App
