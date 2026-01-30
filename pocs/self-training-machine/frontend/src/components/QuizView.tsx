import { useState } from 'react'
import { Quiz, QuizResult } from '../types'

interface QuizViewProps {
  quiz: Quiz
  result: QuizResult | null
  onSubmit: (answers: number[]) => void
  onRetry: () => void
}

function QuizView({ quiz, result, onSubmit, onRetry }: QuizViewProps) {
  const [answers, setAnswers] = useState<number[]>(new Array(quiz.questions.length).fill(-1))
  const [currentQuestion, setCurrentQuestion] = useState(0)

  const handleAnswerSelect = (questionIndex: number, optionIndex: number) => {
    const newAnswers = [...answers]
    newAnswers[questionIndex] = optionIndex
    setAnswers(newAnswers)
  }

  const handleSubmit = () => {
    onSubmit(answers)
  }

  const allAnswered = answers.every(a => a !== -1)
  const question = quiz.questions[currentQuestion]

  if (result) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800 flex items-center justify-center p-6">
        <div className="max-w-2xl w-full bg-white/10 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-white/20 text-center">
          <div
            className={`w-32 h-32 mx-auto mb-6 rounded-full flex items-center justify-center ${
              result.passed ? 'bg-green-500/30' : 'bg-red-500/30'
            }`}
          >
            <span className="text-6xl">{result.passed ? '✓' : '✗'}</span>
          </div>

          <h2 className="text-3xl font-bold text-white mb-4">
            {result.passed ? 'Congratulations!' : 'Keep Learning!'}
          </h2>

          <div className="text-6xl font-bold text-white mb-2">
            {result.percentage.toFixed(0)}%
          </div>
          <p className="text-purple-200 text-xl mb-6">
            {result.score} out of {result.total} correct
          </p>

          <div className="w-full bg-white/20 rounded-full h-4 mb-8">
            <div
              className={`h-4 rounded-full transition-all ${
                result.passed ? 'bg-green-500' : 'bg-red-500'
              }`}
              style={{ width: `${result.percentage}%` }}
            />
          </div>

          {result.passed ? (
            <p className="text-green-300 mb-8">
              You passed! A certificate will be generated.
            </p>
          ) : (
            <div>
              <p className="text-red-300 mb-6">
                You need at least 70% to pass. Review the training and try again.
              </p>
              <button
                onClick={onRetry}
                className="px-8 py-3 bg-purple-500 text-white font-bold rounded-xl hover:bg-purple-600 transition-all"
              >
                Retry Quiz
              </button>
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800 flex flex-col p-6">
      <header className="max-w-4xl mx-auto w-full mb-8">
        <h1 className="text-3xl font-bold text-white mb-4">Knowledge Quiz</h1>
        <div className="flex items-center gap-4">
          <div className="flex-1 bg-white/20 rounded-full h-2">
            <div
              className="bg-purple-500 h-2 rounded-full transition-all"
              style={{ width: `${((currentQuestion + 1) / quiz.questions.length) * 100}%` }}
            />
          </div>
          <span className="text-purple-200 text-sm">
            {currentQuestion + 1} / {quiz.questions.length}
          </span>
        </div>
      </header>

      <main className="max-w-4xl mx-auto w-full flex-1">
        <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-white/20">
          <div className="mb-8">
            <span className="text-purple-300 text-sm font-medium">
              Question {question.id}
            </span>
            <h2 className="text-2xl font-bold text-white mt-2">{question.question}</h2>
          </div>

          <div className="space-y-4">
            {question.options.map((option, index) => (
              <button
                key={index}
                onClick={() => handleAnswerSelect(currentQuestion, index)}
                className={`w-full p-4 rounded-xl text-left transition-all border-2 ${
                  answers[currentQuestion] === index
                    ? 'bg-purple-500/30 border-purple-400 text-white'
                    : 'bg-white/5 border-transparent text-purple-100 hover:bg-white/10 hover:border-purple-400/50'
                }`}
              >
                <span className="inline-block w-8 h-8 rounded-full bg-white/10 text-center leading-8 mr-3 text-sm">
                  {String.fromCharCode(65 + index)}
                </span>
                {option}
              </button>
            ))}
          </div>
        </div>

        <div className="flex justify-between items-center mt-8">
          <button
            onClick={() => setCurrentQuestion(Math.max(0, currentQuestion - 1))}
            disabled={currentQuestion === 0}
            className="px-6 py-3 bg-white/10 text-white rounded-xl hover:bg-white/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            Previous
          </button>

          <div className="flex gap-2">
            {quiz.questions.map((_, index) => (
              <button
                key={index}
                onClick={() => setCurrentQuestion(index)}
                className={`w-10 h-10 rounded-full text-sm font-medium transition-all ${
                  currentQuestion === index
                    ? 'bg-purple-500 text-white'
                    : answers[index] !== -1
                    ? 'bg-green-500/50 text-white'
                    : 'bg-white/10 text-purple-200 hover:bg-white/20'
                }`}
              >
                {index + 1}
              </button>
            ))}
          </div>

          {currentQuestion < quiz.questions.length - 1 ? (
            <button
              onClick={() => setCurrentQuestion(currentQuestion + 1)}
              className="px-6 py-3 bg-purple-500 text-white font-semibold rounded-xl hover:bg-purple-600 transition-all"
            >
              Next
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!allAnswered}
              className="px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-500 text-white font-semibold rounded-xl hover:from-green-600 hover:to-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              Submit Quiz
            </button>
          )}
        </div>
      </main>
    </div>
  )
}

export default QuizView
