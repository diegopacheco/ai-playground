export interface Topic {
  id: string
  title: string
  content: string
}

export interface TrainingContent {
  title: string
  topics: Topic[]
}

export interface QuizQuestion {
  id: number
  question: string
  options: string[]
  correct_index: number
}

export interface Quiz {
  questions: QuizQuestion[]
}

export interface QuizResult {
  score: number
  total: number
  percentage: number
  passed: boolean
}

export interface Certificate {
  id: string
  user_name: string
  training_title: string
  score: number
  total: number
  percentage: number
  date: string
}

export interface SseEventStart {
  type: 'start'
  message: string
  total_steps: number
}

export interface SseEventProgress {
  type: 'progress'
  step: number
  message: string
}

export interface SseEventTrainingReady {
  type: 'training_ready'
  training: TrainingContent
}

export interface SseEventQuizReady {
  type: 'quiz_ready'
  quiz: Quiz
}

export interface SseEventError {
  type: 'error'
  message: string
}

export type SseEvent =
  | SseEventStart
  | SseEventProgress
  | SseEventTrainingReady
  | SseEventQuizReady
  | SseEventError

export type AppView = 'prompt' | 'training' | 'quiz' | 'certificate'

export interface AppState {
  view: AppView
  prompt: string
  training: TrainingContent | null
  quiz: Quiz | null
  quizResult: QuizResult | null
  certificate: Certificate | null
  loading: boolean
  error: string | null
  progress: { step: number; message: string; total: number }
}
