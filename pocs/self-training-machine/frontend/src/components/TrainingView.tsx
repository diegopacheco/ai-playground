import { useState } from 'react'
import { TrainingContent } from '../types'

interface TrainingViewProps {
  training: TrainingContent
  onComplete: () => void
}

function TrainingView({ training, onComplete }: TrainingViewProps) {
  const [selectedTopicId, setSelectedTopicId] = useState(training.topics[0]?.id || '')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [askingQuestion, setAskingQuestion] = useState(false)

  const selectedTopic = training.topics.find(t => t.id === selectedTopicId)

  const handleAskQuestion = async () => {
    if (!question.trim()) return

    setAskingQuestion(true)
    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question.trim(),
          context: selectedTopic?.content || '',
        }),
      })
      const data = await response.json()
      setAnswer(data.answer)
    } catch {
      setAnswer('Sorry, there was an error processing your question.')
    }
    setAskingQuestion(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800 flex flex-col">
      <header className="bg-white/10 backdrop-blur-lg border-b border-white/20 p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <h1 className="text-2xl font-bold text-white">{training.title}</h1>
          <button
            onClick={onComplete}
            className="px-6 py-2 bg-gradient-to-r from-green-500 to-emerald-500 text-white font-semibold rounded-lg hover:from-green-600 hover:to-emerald-600 transition-all shadow-lg"
          >
            Take Quiz
          </button>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        <aside className="w-72 bg-white/5 border-r border-white/10 overflow-y-auto">
          <nav className="p-4">
            <h2 className="text-purple-300 font-semibold mb-4 text-sm uppercase tracking-wider">
              Topics
            </h2>
            <ul className="space-y-2">
              {training.topics.map(topic => (
                <li key={topic.id}>
                  <button
                    onClick={() => setSelectedTopicId(topic.id)}
                    className={`w-full text-left p-3 rounded-lg transition-all ${
                      selectedTopicId === topic.id
                        ? 'bg-purple-500/30 text-white border border-purple-400/50'
                        : 'text-purple-200 hover:bg-white/10'
                    }`}
                  >
                    <span className="text-purple-400 mr-2">{topic.id}.</span>
                    {topic.title}
                  </button>
                </li>
              ))}
            </ul>
          </nav>
        </aside>

        <main className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto p-8">
            {selectedTopic && (
              <article className="max-w-3xl mx-auto">
                <h2 className="text-3xl font-bold text-white mb-6">{selectedTopic.title}</h2>
                <div className="prose prose-invert prose-lg">
                  {selectedTopic.content.split('\n').map((paragraph, i) => (
                    <p key={i} className="text-purple-100 leading-relaxed mb-4">
                      {paragraph}
                    </p>
                  ))}
                </div>
              </article>
            )}
          </div>

          <div className="border-t border-white/20 bg-white/5 p-4">
            <div className="max-w-3xl mx-auto">
              <h3 className="text-white font-semibold mb-3">Ask a Question</h3>
              <div className="flex gap-3">
                <input
                  type="text"
                  value={question}
                  onChange={e => setQuestion(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleAskQuestion()}
                  placeholder="Type your question about this content..."
                  className="flex-1 p-3 rounded-lg bg-white/10 text-white placeholder-purple-300/50 border border-white/20 focus:border-purple-400 focus:outline-none"
                />
                <button
                  onClick={handleAskQuestion}
                  disabled={askingQuestion || !question.trim()}
                  className="px-6 py-3 bg-purple-500 text-white font-semibold rounded-lg hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  {askingQuestion ? 'Asking...' : 'Ask'}
                </button>
              </div>
              {answer && (
                <div className="mt-4 p-4 bg-white/10 rounded-lg border border-white/20">
                  <p className="text-purple-100">{answer}</p>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

export default TrainingView
