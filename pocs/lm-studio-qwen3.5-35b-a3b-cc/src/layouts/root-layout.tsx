import { Link, Outlet } from '@tanstack/react-router'

export function RootLayout() {
  return (
    <div style={{ fontFamily: 'system-ui', maxWidth: '600px', margin: '0 auto', padding: '20px' }}>
      <h1>Paper Rock Cissors</h1>
      <nav>
        <Link to="/" style={{ marginRight: '15px' }}>Play</Link>
        <Link to="/results">Results</Link>
      </nav>
      <Outlet />
    </div>
  )
}
