import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider, createRouter, createRootRoute, createRoute, Outlet, Link } from '@tanstack/react-router'
import './index.css'
import App from './App.jsx'
import Cards from './Cards.jsx'
import Pokedex from './Pokedex.jsx'
import Battle from './Battle.jsx'
import History from './History.jsx'

const queryClient = new QueryClient()

function Layout() {
  return (
    <div className="app-layout">
      <nav className="tab-nav">
        <Link to="/" className="tab-link" activeProps={{ className: 'tab-link active' }}>Home</Link>
        <Link to="/cards" className="tab-link" activeProps={{ className: 'tab-link active' }}>Cards</Link>
        <Link to="/pokedex" className="tab-link" activeProps={{ className: 'tab-link active' }}>Pokedex</Link>
        <Link to="/battle" className="tab-link" activeProps={{ className: 'tab-link active' }}>Battle</Link>
        <Link to="/history" className="tab-link" activeProps={{ className: 'tab-link active' }}>History</Link>
      </nav>
      <Outlet />
    </div>
  )
}

const rootRoute = createRootRoute({ component: Layout })

const indexRoute = createRoute({ getParentRoute: () => rootRoute, path: '/', component: App })
const cardsRoute = createRoute({ getParentRoute: () => rootRoute, path: '/cards', component: Cards })
const pokedexRoute = createRoute({ getParentRoute: () => rootRoute, path: '/pokedex', component: Pokedex })
const battleRoute = createRoute({ getParentRoute: () => rootRoute, path: '/battle', component: Battle })
const historyRoute = createRoute({ getParentRoute: () => rootRoute, path: '/history', component: History })

const routeTree = rootRoute.addChildren([indexRoute, cardsRoute, pokedexRoute, battleRoute, historyRoute])
const router = createRouter({ routeTree })

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
)
