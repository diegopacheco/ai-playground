import { useState } from 'react'
import {
  createRouter,
  createRootRoute,
  createRoute,
  Link,
  Outlet,
} from '@tanstack/react-router'
import HomePage from './pages/HomePage'
import SimulationPage from './pages/SimulationPage'
import ResultsPage from './pages/ResultsPage'
import AboutPage from './pages/AboutPage'
import './App.css'

function RootLayout() {
  return (
    <div className="app">
      <nav className="navbar">
        <div className="nav-content">
          <Link to="/" className="nav-logo">RetireSmart</Link>
          <div className="nav-links">
            <Link to="/" className="nav-link">Home</Link>
            <Link to="/simulate" className="nav-link">Simulate</Link>
            <Link to="/about" className="nav-link">About</Link>
          </div>
        </div>
      </nav>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  )
}

const rootRoute = createRootRoute({ component: RootLayout })

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: HomePage,
})

const simulateRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/simulate',
  component: function SimulateWrapper() {
    const [results, setResults] = useState(null)
    const [inputData, setInputData] = useState(null)
    if (results) {
      return <ResultsPage results={results} inputData={inputData} onBack={() => setResults(null)} />
    }
    return <SimulationPage onResults={(r, i) => { setResults(r); setInputData(i) }} />
  },
})

const aboutRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/about',
  component: AboutPage,
})

const routeTree = rootRoute.addChildren([indexRoute, simulateRoute, aboutRoute])

export const router = createRouter({ routeTree })
