import { Link, Outlet, useMatchRoute } from '@tanstack/react-router'
import './App.css'

function App() {
  const matchRoute = useMatchRoute()

  const navClass = (path) => {
    const opts = path === '/' ? { to: path, fuzzy: false } : { to: path }
    return matchRoute(opts) ? 'nav-link active' : 'nav-link'
  }

  return (
    <div className="app">
      <nav className="navbar">
        <div className="nav-brand">RetireSmart</div>
        <div className="nav-links">
          <Link to="/" className={navClass('/')}>Home</Link>
          <Link to="/simulate" className={navClass('/simulate')}>Simulate</Link>
          <Link to="/results" className={navClass('/results')}>Results</Link>
          <Link to="/about" className={navClass('/about')}>About</Link>
        </div>
      </nav>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  )
}

export default App
