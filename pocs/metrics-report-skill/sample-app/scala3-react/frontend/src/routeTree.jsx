import { createRootRoute, createRoute } from '@tanstack/react-router'
import App from './App'
import HomePage from './pages/HomePage'
import SimulationPage from './pages/SimulationPage'
import ResultsPage from './pages/ResultsPage'
import AboutPage from './pages/AboutPage'

const rootRoute = createRootRoute({ component: App })

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: HomePage,
})

const simulateRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/simulate',
  component: SimulationPage,
})

const resultsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/results',
  component: ResultsPage,
})

const aboutRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/about',
  component: AboutPage,
})

export const routeTree = rootRoute.addChildren([
  indexRoute,
  simulateRoute,
  resultsRoute,
  aboutRoute,
])
