import { ReactNode, useState } from 'react'

const styles = {
  container: { display: 'flex', flexDirection: 'column' as const, minHeight: '100vh' },
  header: { background: '#161b22', borderBottom: '1px solid #30363d', padding: '16px 24px', display: 'flex', alignItems: 'center', gap: '24px' },
  title: { fontSize: '20px', fontWeight: 600, color: '#f0f6fc' },
  nav: { display: 'flex', gap: '8px' },
  tab: { padding: '8px 16px', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '14px', transition: 'all 0.2s' },
  activeTab: { background: '#238636', color: '#fff' },
  inactiveTab: { background: 'transparent', color: '#8b949e' },
  main: { flex: 1, padding: '24px', maxWidth: '1200px', margin: '0 auto', width: '100%' },
}

interface Props {
  children: (activeTab: string) => ReactNode
}

export function Layout({ children }: Props) {
  const [activeTab, setActiveTab] = useState('tasks')
  const tabs = [
    { id: 'tasks', label: 'Tasks' },
    { id: 'projects', label: 'Projects' },
    { id: 'config', label: 'Config' },
  ]
  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>Agent Learner</h1>
        <nav style={styles.nav}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{ ...styles.tab, ...(activeTab === tab.id ? styles.activeTab : styles.inactiveTab) }}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </header>
      <main style={styles.main}>{children(activeTab)}</main>
    </div>
  )
}
