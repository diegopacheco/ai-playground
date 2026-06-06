import { Link, Outlet } from '@tanstack/react-router'
import { ErrorBoundary } from './ErrorBoundary'

export function Layout() {
  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">Sample Shop</span>
        <nav className="nav">
          <Link to="/" className="navlink">Home</Link>
          <Link to="/catalog" className="navlink">Catalog</Link>
          <Link to="/cart" className="navlink">Cart</Link>
          <Link to="/dashboard" className="navlink">Dashboard</Link>
        </nav>
      </header>
      <main className="content">
        <ErrorBoundary>
          <Outlet />
        </ErrorBoundary>
      </main>
    </div>
  )
}
