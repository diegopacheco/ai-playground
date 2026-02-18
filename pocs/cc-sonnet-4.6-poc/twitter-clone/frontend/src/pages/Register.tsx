import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { apiRegister } from '../api'
import { useAuth } from '../AuthContext'

const s: Record<string, React.CSSProperties> = {
  container: { display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 60 },
  title: { fontSize: 32, fontWeight: 800, marginBottom: 32, color: '#1d9bf0' },
  form: { width: '100%', maxWidth: 360, display: 'flex', flexDirection: 'column', gap: 16 },
  input: {
    background: 'transparent',
    border: '1px solid #536471',
    borderRadius: 8,
    padding: '12px 16px',
    color: '#e7e9ea',
    fontSize: 16,
    outline: 'none',
  },
  btn: {
    background: '#1d9bf0',
    border: 'none',
    borderRadius: 24,
    padding: '12px',
    color: '#fff',
    fontSize: 16,
    fontWeight: 700,
    cursor: 'pointer',
  },
  error: { color: '#f4212e', fontSize: 14, textAlign: 'center' },
  link: { color: '#1d9bf0', textAlign: 'center', marginTop: 8 },
}

export default function Register() {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const navigate = useNavigate()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const data = await apiRegister(username, email, password)
      login(data)
      navigate('/')
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={s.container}>
      <h1 style={s.title}>Join 𝕏 Clone</h1>
      <form onSubmit={handleSubmit} style={s.form}>
        <input
          style={s.input}
          placeholder="Username"
          value={username}
          onChange={e => setUsername(e.target.value)}
          required
        />
        <input
          style={s.input}
          type="email"
          placeholder="Email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
        />
        <input
          style={s.input}
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
          minLength={6}
        />
        {error && <p style={s.error}>{error}</p>}
        <button style={s.btn} type="submit" disabled={loading}>
          {loading ? 'Creating account...' : 'Sign up'}
        </button>
        <p style={s.link}>
          Have an account? <Link to="/login" style={{ color: '#1d9bf0' }}>Log in</Link>
        </p>
      </form>
    </div>
  )
}
