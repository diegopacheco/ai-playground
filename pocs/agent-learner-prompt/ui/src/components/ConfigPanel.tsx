import { useState, useEffect } from 'react'
import { useConfig, useUpdateConfig } from '../api/queries'

const styles = {
  panel: { background: '#161b22', border: '1px solid #30363d', borderRadius: '6px', padding: '24px' },
  title: { fontSize: '18px', fontWeight: 600, color: '#f0f6fc', marginBottom: '20px' },
  field: { marginBottom: '16px' },
  label: { display: 'block', color: '#8b949e', fontSize: '12px', marginBottom: '6px', textTransform: 'uppercase' as const },
  input: { width: '100%', padding: '10px', background: '#0d1117', border: '1px solid #30363d', borderRadius: '6px', color: '#c9d1d9', fontSize: '14px' },
  select: { width: '100%', padding: '10px', background: '#0d1117', border: '1px solid #30363d', borderRadius: '6px', color: '#c9d1d9', fontSize: '14px' },
  button: { padding: '10px 20px', background: '#238636', border: 'none', borderRadius: '6px', color: '#fff', fontSize: '14px', fontWeight: 600, cursor: 'pointer', marginTop: '8px' },
  success: { color: '#3fb950', marginTop: '8px', fontSize: '14px' },
  error: { color: '#f85149', marginTop: '8px', fontSize: '14px' },
}

const AGENTS = ['claude', 'codex', 'copilot', 'gemini']

const MODELS: Record<string, string[]> = {
  claude: ['opus', 'sonnet', 'haiku'],
  codex: ['gpt-5.2-codex', 'gpt-5.2', 'gpt-5.1-codex-max', 'gpt-5.1-codex-mini'],
  copilot: [
    'claude-sonnet-4.5',
    'claude-haiku-4.5',
    'claude-opus-4.5',
    'claude-sonnet-4',
    'gemini-3-pro',
    'gpt-5.2-codex',
    'gpt-5.2',
    'gpt-5.1-codex-max',
    'gpt-5.1-codex',
    'gpt-5.1',
    'gpt-5',
    'gpt-5.1-codex-mini',
    'gpt-5-mini',
    'gpt-4.1',
  ],
  gemini: ['auto-gemini-3', 'gemini-3-pro', 'gemini-3-flash', 'auto-gemini-2.5', 'gemini-2.5-pro', 'gemini-2.5-flash'],
}

export function ConfigPanel() {
  const { data: config, isLoading } = useConfig()
  const mutation = useUpdateConfig()
  const [agent, setAgent] = useState('')
  const [model, setModel] = useState('')
  const [cycles, setCycles] = useState(3)
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
  const handleSave = () => {
    mutation.mutate({ agent, model, cycles })
  }
  const availableModels = MODELS[agent] || []
  if (isLoading) return <div>Loading...</div>
  return (
    <div style={styles.panel}>
      <h2 style={styles.title}>Configuration</h2>
      <div style={styles.field}>
        <label style={styles.label}>Agent</label>
        <select style={styles.select} value={agent} onChange={(e) => handleAgentChange(e.target.value)}>
          {AGENTS.map((a) => <option key={a} value={a}>{a}</option>)}
        </select>
      </div>
      <div style={styles.field}>
        <label style={styles.label}>Model</label>
        <select style={styles.select} value={model} onChange={(e) => setModel(e.target.value)}>
          {availableModels.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
      </div>
      <div style={styles.field}>
        <label style={styles.label}>Cycles (1-10)</label>
        <input
          style={styles.input}
          type="number"
          min={1}
          max={10}
          value={cycles}
          onChange={(e) => setCycles(parseInt(e.target.value) || 3)}
        />
      </div>
      <button style={styles.button} onClick={handleSave} disabled={mutation.isPending}>
        {mutation.isPending ? 'Saving...' : 'Save Configuration'}
      </button>
      {mutation.isSuccess && <div style={styles.success}>Configuration saved</div>}
      {mutation.isError && <div style={styles.error}>{(mutation.error as Error).message}</div>}
    </div>
  )
}
