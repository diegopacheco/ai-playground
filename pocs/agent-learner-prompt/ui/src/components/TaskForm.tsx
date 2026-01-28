import { useState, useEffect } from 'react'
import { useSubmitTask, useConfig } from '../api/queries'

const styles = {
  form: { background: '#161b22', border: '1px solid #30363d', borderRadius: '6px', padding: '20px', marginBottom: '20px' },
  label: { display: 'block', color: '#8b949e', fontSize: '12px', marginBottom: '6px', textTransform: 'uppercase' as const },
  textarea: { width: '100%', padding: '12px', background: '#0d1117', border: '1px solid #30363d', borderRadius: '6px', color: '#c9d1d9', fontSize: '14px', resize: 'vertical' as const, minHeight: '100px' },
  row: { display: 'flex', gap: '16px', marginTop: '16px' },
  select: { flex: 1, padding: '10px', background: '#0d1117', border: '1px solid #30363d', borderRadius: '6px', color: '#c9d1d9', fontSize: '14px' },
  input: { flex: 1, padding: '10px', background: '#0d1117', border: '1px solid #30363d', borderRadius: '6px', color: '#c9d1d9', fontSize: '14px' },
  button: { padding: '10px 20px', background: '#238636', border: 'none', borderRadius: '6px', color: '#fff', fontSize: '14px', fontWeight: 600, cursor: 'pointer' },
  error: { color: '#f85149', marginTop: '8px', fontSize: '14px' },
}

const AGENTS = ['claude', 'codex', 'copilot', 'gemini']

const MODELS: Record<string, string[]> = {
  claude: ['sonnet', 'opus', 'haiku'],
  codex: ['gpt-5.2', 'gpt-4o', 'o3'],
  copilot: ['claude-sonnet-4', 'gpt-4o', 'o3'],
  gemini: ['gemini-2.5-pro', 'gemini-2.5-flash'],
}

interface Props {
  onTaskCreated: (taskId: string) => void
}

export function TaskForm({ onTaskCreated }: Props) {
  const [task, setTask] = useState('')
  const [agent, setAgent] = useState('claude')
  const [model, setModel] = useState('sonnet')
  const [cycles, setCycles] = useState(3)
  const { data: config } = useConfig()
  const mutation = useSubmitTask()
  useEffect(() => {
    if (config) {
      setAgent(config.agent)
      setModel(config.model)
      setCycles(config.cycles)
    }
  }, [config])
  const handleAgentChange = (newAgent: string) => {
    setAgent(newAgent)
    const models = MODELS[newAgent] || []
    if (models.length > 0 && !models.includes(model)) {
      setModel(models[0])
    }
  }
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!task.trim()) return
    mutation.mutate({ task: task.trim(), agent, model, cycles }, {
      onSuccess: (data) => {
        onTaskCreated(data.task_id)
        setTask('')
      },
    })
  }
  const availableModels = MODELS[agent] || []
  return (
    <form style={styles.form} onSubmit={handleSubmit}>
      <label style={styles.label}>Task Description</label>
      <textarea
        style={styles.textarea}
        value={task}
        onChange={(e) => setTask(e.target.value)}
        placeholder="Describe what code you want to generate..."
      />
      <div style={styles.row}>
        <select style={styles.select} value={agent} onChange={(e) => handleAgentChange(e.target.value)}>
          {AGENTS.map((a) => <option key={a} value={a}>{a}</option>)}
        </select>
        <select style={styles.select} value={model} onChange={(e) => setModel(e.target.value)}>
          {availableModels.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
        <input style={styles.input} type="number" min={1} max={10} value={cycles} onChange={(e) => setCycles(parseInt(e.target.value) || 3)} />
        <button style={styles.button} type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? 'Starting...' : 'Run Task'}
        </button>
      </div>
      {mutation.isError && <div style={styles.error}>{(mutation.error as Error).message}</div>}
    </form>
  )
}
