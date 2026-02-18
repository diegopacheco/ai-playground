import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../AuthContext'

const s: Record<string, React.CSSProperties> = {
  nav: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px 0',
    borderBottom: '1px solid #2f3336',
    marginBottom: 16,
    position: 'sticky',
    top: 0,
    background: 'rgba(0,0,0,0.85)',
    backdropFilter: 'blur(12px)',
    zIndex: 10,
  },
  brand: { fontSize: 22, fontWeight: 800, color: '#1d9bf0', textDecoration: 'none' },
  links: { display: 'flex', gap: 20, alignItems: 'center' },
  link: { color: '#e7e9ea', textDecoration: 'none', fontSize: 15, fontWeight: 500 },
  btn: {
    background: 'transparent',
    border: '1px solid #536471',
    color: '#e7e9ea',
    borderRadius: 20,
    padding: '6px 16px',
    cursor: 'pointer',
    fontSize: 14,
  },
}

export default function Navbar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <nav style={s.nav}>
      <Link to="/" style={s.brand}>𝕏 Clone</Link>
      <div style={s.links}>
        <Link to="/" style={s.link}>Home</Link>
        <Link to="/search" style={s.link}>Search</Link>
        {user && (
          <Link to={`/profile/${user.user_id}`} style={s.link}>@{user.username}</Link>
        )}
        <button onClick={handleLogout} style={s.btn}>Logout</button>
      </div>
    </nav>
  )
}
