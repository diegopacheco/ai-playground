import { useState } from 'react'
import { useSubmitTask, useConfig } from '../api/queries'

const styles = {
  form: { background: '#161b22', border: '1px solid #30363d', borderRadius: '6px', padding: '20px', marginBottom: '20px' },
  label: { display: 'block', color: '#8b949e', fontSize: '12px', marginBottom: '6px', textTransform: 'uppercase' as const },
  textarea: { width: '100%', padding: '12px', background: '#0d1117', border: '1px solid #30363d', borderRadius: '6px', color: '#c9d1d9', fontSize: '14px', resize: 'vertical' as const, minHeight: '100px' },
  row: { display: 'flex', gap: '16px', marginTop: '16px' },
  input: { flex: 1, padding: '10px', background: '#0d1117', border: '1px solid #30363d', borderRadius: '6px', color: '#c9d1d9', fontSize: '14px' },
  button: { padding: '10px 20px', background: '#238636', border: 'none', borderRadius: '6px', color: '#fff', fontSize: '14px', fontWeight: 600, cursor: 'pointer' },
  error: { color: '#f85149', marginTop: '8px', fontSize: '14px' },
}

interface Props {
  onTaskCreated: (taskId: string) => void
}

export function TaskForm({ onTaskCreated }: Props) {
  const [task, setTask] = useState('')
  const { data: config } = useConfig()
  const mutation = useSubmitTask()
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!task.trim()) return
    mutation.mutate({ task: task.trim() }, {
      onSuccess: (data) => {
        onTaskCreated(data.task_id)
        setTask('')
      },
    })
  }
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
        <input style={styles.input} value={config?.agent || 'claude'} readOnly placeholder="Agent" />
        <input style={styles.input} value={config?.model || ''} readOnly placeholder="Model" />
        <input style={styles.input} value={config?.cycles || 3} readOnly placeholder="Cycles" />
        <button style={styles.button} type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? 'Starting...' : 'Run Task'}
        </button>
      </div>
      {mutation.isError && <div style={styles.error}>{(mutation.error as Error).message}</div>}
    </form>
  )
}
