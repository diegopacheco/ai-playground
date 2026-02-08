import { createRouter, createRoute, createRootRoute } from '@tanstack/react-router'
import App from './App'
import SetupPage from './pages/SetupPage'
import CanvasPage from './pages/CanvasPage'
import ProgressPage from './pages/ProgressPage'
import PreviewPage from './pages/PreviewPage'
import HistoryPage from './pages/HistoryPage'

const rootRoute = createRootRoute({
  component: App,
})

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: SetupPage,
})

const canvasRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/canvas/$projectId',
  component: CanvasPage,
})

const progressRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/progress/$projectId',
  component: ProgressPage,
})

const previewRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/preview/$projectId',
  component: PreviewPage,
})

const historyRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/history',
  component: HistoryPage,
})

const routeTree = rootRoute.addChildren([
  indexRoute,
  canvasRoute,
  progressRoute,
  previewRoute,
  historyRoute,
])

export const router = createRouter({ routeTree })

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
