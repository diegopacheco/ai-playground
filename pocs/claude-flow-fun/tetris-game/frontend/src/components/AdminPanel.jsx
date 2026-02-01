import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

const AdminPanel = () => {
  const navigate = useNavigate()
  const [config, setConfig] = useState({
    theme: 'dark',
    boardWidth: 10,
    boardHeight: 20,
    growInterval: 30,
    dropSpeed: 1000,
    freezeChance: 2
  })
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => {
    fetch('/api/config')
      .then(res => res.json())
      .then(data => {
        if (data) {
          setConfig({
            theme: data.theme || 'dark',
            boardWidth: data.board_width || 10,
            boardHeight: data.board_height || 20,
            growInterval: data.grow_interval ? data.grow_interval / 1000 : 30,
            dropSpeed: data.drop_speed || 1000,
            freezeChance: data.freeze_chance || 2
          })
        }
      })
      .catch(() => {})
  }, [])

  const handleChange = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }))
  }

  const handleSave = async () => {
    setSaving(true)
    setMessage('')
    const payload = {
      theme: config.theme,
      board_width: config.boardWidth,
      board_height: config.boardHeight,
      grow_interval: config.growInterval * 1000,
      drop_speed: config.dropSpeed,
      freeze_chance: config.freezeChance,
      level_time_multiplier: 0.85
    }
    try {
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      if (res.ok) {
        setMessage('Configuration saved successfully!')
      } else {
        setMessage('Failed to save configuration')
      }
    } catch (err) {
      setMessage('Error connecting to server')
    }
    setSaving(false)
    setTimeout(() => setMessage(''), 3000)
  }

  const themes = [
    { value: 'dark', label: 'Dark Mode' },
    { value: 'light', label: 'Light Mode' },
    { value: 'retro', label: 'Retro Green' },
    { value: 'neon', label: 'Neon Purple' },
    { value: 'ocean', label: 'Ocean Blue' }
  ]

  return (
    <div className="admin-container">
      <div className="admin-header">
        <h1>Admin Panel</h1>
        <button onClick={() => navigate('/')} className="back-btn">
          Back to Game
        </button>
      </div>
      <div className="admin-content">
        <section className="config-section">
          <h2>Theme Settings</h2>
          <div className="theme-selector">
            {themes.map(theme => (
              <button
                key={theme.value}
                className={`theme-btn theme-${theme.value} ${config.theme === theme.value ? 'active' : ''}`}
                onClick={() => handleChange('theme', theme.value)}
              >
                {theme.label}
              </button>
            ))}
          </div>
        </section>
        <section className="config-section">
          <h2>Board Settings</h2>
          <div className="slider-group">
            <label>
              Initial Board Width: {config.boardWidth}
            </label>
            <input
              type="range"
              min="8"
              max="16"
              value={config.boardWidth}
              onChange={(e) => handleChange('boardWidth', parseInt(e.target.value))}
            />
          </div>
          <div className="slider-group">
            <label>
              Initial Board Height: {config.boardHeight}
            </label>
            <input
              type="range"
              min="15"
              max="30"
              value={config.boardHeight}
              onChange={(e) => handleChange('boardHeight', parseInt(e.target.value))}
            />
          </div>
        </section>
        <section className="config-section">
          <h2>Time Settings</h2>
          <div className="slider-group">
            <label>
              Board Grow Interval: {config.growInterval} seconds
            </label>
            <input
              type="range"
              min="10"
              max="120"
              step="5"
              value={config.growInterval}
              onChange={(e) => handleChange('growInterval', parseInt(e.target.value))}
            />
          </div>
          <div className="slider-group">
            <label>
              Drop Speed: {config.dropSpeed}ms
            </label>
            <input
              type="range"
              min="100"
              max="2000"
              step="100"
              value={config.dropSpeed}
              onChange={(e) => handleChange('dropSpeed', parseInt(e.target.value))}
            />
          </div>
          <div className="slider-group">
            <label>
              Freeze Bonus Chance: {config.freezeChance}%
            </label>
            <input
              type="range"
              min="0"
              max="10"
              value={config.freezeChance}
              onChange={(e) => handleChange('freezeChance', parseInt(e.target.value))}
            />
          </div>
        </section>
        <div className="save-section">
          <button
            onClick={handleSave}
            disabled={saving}
            className="save-btn"
          >
            {saving ? 'Saving...' : 'Save Configuration'}
          </button>
          {message && (
            <p className={`message ${message.includes('success') ? 'success' : 'error'}`}>
              {message}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

export default AdminPanel
