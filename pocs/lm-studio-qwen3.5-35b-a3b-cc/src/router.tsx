import { createRouter } from '@tanstack/react-router'
import { Route as rootRoute } from './routes/__root__'
import { Route as indexRoute } from './routes/index'
import { Route as resultsRoute } from './routes/results'

const routeTree = rootRoute.addChildren([indexRoute, resultsRoute])

export const router = createRouter({ routeTree })

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
