import { useState } from 'react'
import ClusterPage from './pages/ClusterPage'
import LogsPage from './pages/LogsPage'
import FixPage from './pages/FixPage'
import HistoryPage from './pages/HistoryPage'

const tabs = ['Cluster', 'Logs', 'Fix', 'History'] as const

export default function App() {
  const [active, setActive] = useState<typeof tabs[number]>('Cluster')

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', background: '#0d1117', color: '#c9d1d9', minHeight: '100vh' }}>
      <header style={{ background: '#161b22', borderBottom: '1px solid #30363d', padding: '12px 24px', display: 'flex', alignItems: 'center', gap: 24 }}>
        <h1 style={{ margin: 0, fontSize: 20, color: '#58a6ff' }}>Kovalski</h1>
        <nav style={{ display: 'flex', gap: 4 }}>
          {tabs.map(tab => (
            <button
              key={tab}
              onClick={() => setActive(tab)}
              style={{
                padding: '8px 16px',
                background: active === tab ? '#21262d' : 'transparent',
                color: active === tab ? '#58a6ff' : '#8b949e',
                border: active === tab ? '1px solid #30363d' : '1px solid transparent',
                borderRadius: 6,
                cursor: 'pointer',
                fontSize: 14,
                fontWeight: active === tab ? 600 : 400,
              }}
            >
              {tab}
            </button>
          ))}
        </nav>
      </header>
      <main style={{ padding: 24 }}>
        {active === 'Cluster' && <ClusterPage />}
        {active === 'Logs' && <LogsPage />}
        {active === 'Fix' && <FixPage />}
        {active === 'History' && <HistoryPage />}
      </main>
    </div>
  )
}
