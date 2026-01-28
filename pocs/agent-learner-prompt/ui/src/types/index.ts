export interface TaskRequest {
  task: string
  agent?: string
  model?: string
  cycles?: number
}

export interface TaskResponse {
  task_id: string
  status: string
}

export interface TaskStatus {
  task_id: string
  status: string
  current_cycle: number
  total_cycles: number
  phase: string
  completed: boolean
  success: boolean
}

export interface ProjectInfo {
  name: string
  cycles: string[]
  has_memory: boolean
  has_mistakes: boolean
}

export interface ProjectDetail {
  name: string
  memory: string
  mistakes: string
  prompts: string
  cycles: CycleInfo[]
}

export interface CycleInfo {
  cycle_number: number
  has_prompt: boolean
  has_output: boolean
  has_review: boolean
}

export interface ConfigRequest {
  agent?: string
  model?: string
  cycles?: number
}

export interface ConfigResponse {
  agent: string
  model: string
  cycles: number
}

export interface ProgressEvent {
  task_id: string
  event_type: string
  cycle: number
  phase: string
  message: string
}
