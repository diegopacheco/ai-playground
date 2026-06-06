import { createRouter, createRoute, createRootRoute } from '@tanstack/react-router'
import { Layout } from './components/Layout'
import { HomePage } from './features/home/routes/home'
import { CatalogPage } from './features/catalog/routes/catalog'
import { CartPage } from './features/cart/routes/cart'
import { DashboardPage } from './features/dashboard/routes/dashboard'

const rootRoute = createRootRoute({ component: Layout })

const homeRoute = createRoute({ getParentRoute: () => rootRoute, path: '/', component: HomePage })
const catalogRoute = createRoute({ getParentRoute: () => rootRoute, path: '/catalog', component: CatalogPage })
const cartRoute = createRoute({ getParentRoute: () => rootRoute, path: '/cart', component: CartPage })
const dashboardRoute = createRoute({ getParentRoute: () => rootRoute, path: '/dashboard', component: DashboardPage })

const routeTree = rootRoute.addChildren([homeRoute, catalogRoute, cartRoute, dashboardRoute])

export const router = createRouter({ routeTree })

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
