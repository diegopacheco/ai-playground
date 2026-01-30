import { useState } from 'react'
import { AppState, SseEvent, QuizResult, Certificate } from './types'
import PromptView from './components/PromptView'
import TrainingView from './components/TrainingView'
import QuizView from './components/QuizView'
import CertificateView from './components/CertificateView'
import LoadingView from './components/LoadingView'

const initialState: AppState = {
  view: 'prompt',
  prompt: '',
  training: null,
  quiz: null,
  quizResult: null,
  certificate: null,
  loading: false,
  error: null,
  progress: { step: 0, message: '', total: 3 },
}

function App() {
  const [state, setState] = useState<AppState>(initialState)

  const handlePromptSubmit = async (prompt: string) => {
    setState(prev => ({
      ...prev,
      prompt,
      loading: true,
      error: null,
      progress: { step: 0, message: 'Starting...', total: 3 },
    }))

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      })

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            try {
              const event: SseEvent = JSON.parse(data)
              handleSseEvent(event)
            } catch {
              continue
            }
          }
        }
      }
    } catch (err) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Unknown error',
      }))
    }
  }

  const handleSseEvent = (event: SseEvent) => {
    switch (event.type) {
      case 'start':
        setState(prev => ({
          ...prev,
          progress: { step: 0, message: event.message, total: event.total_steps },
        }))
        break
      case 'progress':
        setState(prev => ({
          ...prev,
          progress: { ...prev.progress, step: event.step, message: event.message },
        }))
        break
      case 'training_ready':
        setState(prev => ({
          ...prev,
          training: event.training,
        }))
        break
      case 'quiz_ready':
        setState(prev => ({
          ...prev,
          quiz: event.quiz,
          loading: false,
          view: 'training',
        }))
        break
      case 'error':
        setState(prev => ({
          ...prev,
          loading: false,
          error: event.message,
        }))
        break
    }
  }

  const handleTrainingComplete = () => {
    setState(prev => ({ ...prev, view: 'quiz' }))
  }

  const handleQuizSubmit = async (answers: number[]) => {
    try {
      const response = await fetch('/api/submit-quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers }),
      })
      const result: QuizResult = await response.json()
      setState(prev => ({ ...prev, quizResult: result }))

      if (result.passed) {
        const userName = prompt('Enter your name for the certificate:') || 'Student'
        const certResponse = await fetch('/api/certificate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_name: userName,
            training_title: state.training?.title || 'Training',
            score: result.score,
            total: result.total,
            percentage: result.percentage,
          }),
        })
        const certificate: Certificate = await certResponse.json()
        setState(prev => ({ ...prev, certificate, view: 'certificate' }))
      }
    } catch (err) {
      setState(prev => ({
        ...prev,
        error: err instanceof Error ? err.message : 'Failed to submit quiz',
      }))
    }
  }

  const handleRestart = () => {
    setState(initialState)
  }

  if (state.loading) {
    return <LoadingView progress={state.progress} />
  }

  if (state.error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-red-900 to-red-700 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-md w-full text-center">
          <h2 className="text-2xl font-bold text-red-600 mb-4">Error</h2>
          <p className="text-gray-700 mb-6">{state.error}</p>
          <button
            onClick={handleRestart}
            className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800">
      {state.view === 'prompt' && <PromptView onSubmit={handlePromptSubmit} />}
      {state.view === 'training' && state.training && (
        <TrainingView training={state.training} onComplete={handleTrainingComplete} />
      )}
      {state.view === 'quiz' && state.quiz && (
        <QuizView
          quiz={state.quiz}
          result={state.quizResult}
          onSubmit={handleQuizSubmit}
          onRetry={() => setState(prev => ({ ...prev, quizResult: null }))}
        />
      )}
      {state.view === 'certificate' && state.certificate && (
        <CertificateView certificate={state.certificate} onRestart={handleRestart} />
      )}
    </div>
  )
}

export default App
