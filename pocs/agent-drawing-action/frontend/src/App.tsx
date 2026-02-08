import { Outlet, Link } from '@tanstack/react-router'

const appStyle: React.CSSProperties = {
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  minHeight: '100vh',
  background: '#0f0f0f',
  color: '#e0e0e0',
}

const navStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '24px',
  padding: '12px 24px',
  background: '#1a1a2e',
  borderBottom: '1px solid #2a2a4a',
}

const logoStyle: React.CSSProperties = {
  fontSize: '18px',
  fontWeight: 700,
  color: '#7c4dff',
  textDecoration: 'none',
}

const linkStyle: React.CSSProperties = {
  color: '#9e9e9e',
  textDecoration: 'none',
  fontSize: '14px',
}

export default function App() {
  return (
    <div style={appStyle}>
      <nav style={navStyle}>
        <Link to="/" style={logoStyle}>Agent Drawing Action</Link>
        <Link to="/" style={linkStyle}>Draw</Link>
        <Link to="/history" style={linkStyle}>History</Link>
      </nav>
      <Outlet />
    </div>
  )
}
