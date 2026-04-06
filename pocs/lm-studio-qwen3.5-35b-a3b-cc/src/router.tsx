import { createRouter } from '@tanstack/react-router'
import { route as rootRoute } from './routes/_root'
import { route as index } from './routes/_root/index'
import { route as results } from './routes/_root/results'

export const router = createRouter({
  routeTree: rootRoute.addChildren([index, results]),
  defaultPreload: 'intent',
  scrollRestoration: false,
})

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
