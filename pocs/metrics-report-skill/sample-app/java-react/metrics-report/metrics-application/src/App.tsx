import { useState, useEffect } from 'react'
import { Routes, Route, NavLink, Navigate } from 'react-router-dom'
import { MetricsReport, TEST_TYPES, TestType } from './types/metrics'
import SearchBar from './components/SearchBar'
import Dashboard from './pages/Dashboard'
import Tests from './pages/Tests'
import Coverage from './pages/Coverage'
import Failures from './pages/Failures'
import Authors from './pages/Authors'
import Trends from './pages/Trends'
import Quality from './pages/Quality'

function normalizeMetrics(raw: any): MetricsReport {
  const byType: any = {}
  for (const t of TEST_TYPES) {
    if (raw.tests[t]) {
      const typeData = raw.tests[t]
      const files = (typeData.files || []).map((f: any) => {
        const tests = (f.tests || []).map((tc: any) => ({
          name: tc.name,
          status: tc.status,
          duration: tc.duration || 0,
          line: tc.line || 0,
          githubUrl: tc.githubUrl || f.githubUrl || '',
        }))
        const passing = tests.filter((tc: any) => tc.status === 'pass').length
        const failing = tests.filter((tc: any) => tc.status === 'fail').length
        return {
          path: f.file || f.path || '',
          type: t as TestType,
          testCount: tests.length,
          passing,
          failing,
          author: f.author || 'unknown',
          githubUrl: f.githubUrl || '',
          tests,
        }
      })
      byType[t] = {
        total: typeData.total || 0,
        passing: typeData.passing || 0,
        failing: typeData.failing || 0,
        duration: typeData.duration || 0,
        files,
      }
    }
  }
  return {
    ...raw,
    tests: {
      total: raw.tests.total || 0,
      passing: raw.tests.passing || 0,
      failing: raw.tests.failing || 0,
      byType,
    },
  }
}

function App() {
  const [data, setData] = useState<MetricsReport | null>(null)
  const [history, setHistory] = useState<MetricsReport[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('theme') === 'dark'
  })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light')
    localStorage.setItem('theme', darkMode ? 'dark' : 'light')
  }, [darkMode])

  useEffect(() => {
    async function loadData() {
      try {
        const res = await fetch('/data/metrics-latest.json')
        if (!res.ok) throw new Error('Failed to load metrics data')
        const raw = await res.json()
        const metrics = raw.tests?.byType ? raw : normalizeMetrics(raw)
        setData(metrics)

        try {
          const indexRes = await fetch('/data/history-index.json')
          if (indexRes.ok) {
            const files: string[] = await indexRes.json()
            const historyData = await Promise.all(
              files.slice(0, 50).map(async (f) => {
                const r = await fetch(`/data/history/${f}`)
                const raw = await r.json()
                return raw.tests?.byType ? raw : normalizeMetrics(raw)
              })
            )
            setHistory(historyData)
          }
        } catch {
          setHistory([])
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  if (loading) {
    return <div className="loading-screen"><div className="spinner" /><p>Loading metrics...</p></div>
  }

  if (error || !data) {
    return <div className="error-screen"><h2>Error</h2><p>{error || 'No data available'}</p><p>Run the metrics-report skill first to generate data.</p></div>
  }

  const navItems = [
    { path: '/dashboard', label: 'Dashboard', icon: 'D' },
    { path: '/tests', label: 'Tests', icon: 'T' },
    { path: '/coverage', label: 'Coverage', icon: 'C' },
    { path: '/failures', label: 'Failures', icon: 'F' },
    { path: '/authors', label: 'Authors', icon: 'A' },
    { path: '/trends', label: 'Trends', icon: 'R' },
    { path: '/quality', label: 'Quality', icon: 'Q' },
  ]

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>Metrics</h1>
          <span className="version">v1.0</span>
        </div>
        <nav className="sidebar-nav">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <button className="theme-toggle" onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? 'Light Mode' : 'Dark Mode'}
          </button>
        </div>
      </aside>
      <main className="main-content">
        <header className="top-bar">
          <SearchBar value={searchQuery} onChange={setSearchQuery} placeholder="Search tests, files, authors, metrics..." />
          <div className="repo-info">
            <span>{data.repository.name}</span>
            <span className="branch">{data.repository.branch}</span>
            <span className="commit">{data.repository.commit}</span>
          </div>
        </header>
        <div className="page-content">
          <Routes>
            <Route path="/dashboard" element={<Dashboard data={data} />} />
            <Route path="/tests" element={<Tests data={data} searchQuery={searchQuery} />} />
            <Route path="/coverage" element={<Coverage data={data} searchQuery={searchQuery} />} />
            <Route path="/failures" element={<Failures data={data} searchQuery={searchQuery} history={history} />} />
            <Route path="/authors" element={<Authors data={data} searchQuery={searchQuery} />} />
            <Route path="/trends" element={<Trends history={history.length > 0 ? history : [data]} />} />
            <Route path="/quality" element={<Quality data={data} />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

export default App
