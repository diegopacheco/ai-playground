import { createRootRoute, Link, Outlet } from '@tanstack/react-router'

export const Route = createRootRoute({
  component: () => (
    <>
      <nav style={{ padding: '1rem', display: 'flex', gap: '1rem' }}>
        <Link to="/">Play</Link>
        <Link to="/history">History</Link>
      </nav>
      <Outlet />
    </>
  ),
})
