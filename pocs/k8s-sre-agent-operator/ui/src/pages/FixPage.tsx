import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { postFix } from '../api'
import type { FixResult } from '../api'

export default function FixPage() {
  const [result, setResult] = useState<FixResult | null>(null)

  const mutation = useMutation({
    mutationFn: postFix,
    onSuccess: (data) => setResult(data),
  })

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, color: '#58a6ff' }}>Fix Broken Deployments</h2>
        <button
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending}
          style={{
            padding: '8px 20px',
            background: mutation.isPending ? '#30363d' : '#238636',
            color: '#ffffff',
            border: 'none',
            borderRadius: 6,
            cursor: mutation.isPending ? 'not-allowed' : 'pointer',
            fontSize: 14,
            fontWeight: 600,
          }}
        >
          {mutation.isPending ? 'Fixing... (this may take a while)' : 'Run Fix'}
        </button>
      </div>

      {mutation.isError && (
        <div style={{ background: '#3d1518', border: '1px solid #da3633', borderRadius: 8, padding: 16, marginBottom: 16 }}>
          <p style={{ color: '#da3633', fontWeight: 600 }}>Error</p>
          <pre style={{ fontSize: 12, marginTop: 8, color: '#f85149' }}>{(mutation.error as Error).message}</pre>
        </div>
      )}

      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ background: result.success ? '#0d2818' : '#3d1518', border: `1px solid ${result.success ? '#238636' : '#da3633'}`, borderRadius: 8, padding: 12 }}>
            <span style={{ fontWeight: 600, color: result.success ? '#3fb950' : '#f85149' }}>
              {result.success ? 'Fix applied successfully' : 'Fix failed'}
            </span>
          </div>

          {result.diagnostics && (
            <Section title="Diagnostics" content={result.diagnostics} />
          )}
          {result.claude_response && (
            <Section title="Claude Analysis" content={result.claude_response} />
          )}
          {result.kubectl_output && (
            <Section title="kubectl Output" content={result.kubectl_output} />
          )}
        </div>
      )}
    </div>
  )
}

function Section({ title, content }: { title: string; content: string }) {
  return (
    <div>
      <h3 style={{ fontSize: 14, color: '#8b949e', marginBottom: 8 }}>{title}</h3>
      <pre style={{
        background: '#161b22',
        border: '1px solid #30363d',
        borderRadius: 8,
        padding: 16,
        fontSize: 12,
        fontFamily: 'monospace',
        overflow: 'auto',
        maxHeight: 400,
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
      }}>
        {content}
      </pre>
    </div>
  )
}
