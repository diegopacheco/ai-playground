import { createRouter, createRoute, createRootRoute } from '@tanstack/react-router'
import { Layout } from './components/Layout'
import { HomePage } from './pages/HomePage'
import { ListPage } from './pages/ListPage'

const rootRoute = createRootRoute({ component: Layout })

const homeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: HomePage,
})

const listRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/list/$listId',
  component: ListPage,
})

const routeTree = rootRoute.addChildren([homeRoute, listRoute])

export const router = createRouter({ routeTree })

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
