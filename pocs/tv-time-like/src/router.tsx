import { createRootRoute, createRoute, createRouter } from "@tanstack/react-router"
import { AppShell } from "./components/AppShell"
import { LibraryPage } from "./pages/LibraryPage"
import { MetricsPage } from "./pages/MetricsPage"
import { SearchPage } from "./pages/SearchPage"

const rootRoute = createRootRoute({ component: AppShell })
const searchRoute = createRoute({ getParentRoute: () => rootRoute, path: "/", component: SearchPage })
const libraryRoute = createRoute({ getParentRoute: () => rootRoute, path: "/library", component: LibraryPage })
const metricsRoute = createRoute({ getParentRoute: () => rootRoute, path: "/metrics", component: MetricsPage })
const routeTree = rootRoute.addChildren([searchRoute, libraryRoute, metricsRoute])

export const router = createRouter({ routeTree })

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router
  }
}
