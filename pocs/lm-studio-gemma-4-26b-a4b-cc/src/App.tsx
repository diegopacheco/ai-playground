
import { createRootRoute, createRoute, createRouter, RouterProvider } from '@tanstack/react-router'
import React from 'react'
import './index.css'

import { GamePage } from './pages/GamePage'
import { HistoryPage } from './pages/HistoryPage'

const rootRoute = createRootRoute()

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: GamePage,
})

const historyRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/history',
  component: HistoryPage,
})

const routeTree = rootRoute.addChildren([indexRoute, historyRoute])

const router = createRouter({ routeTree })

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

export function App() {
  return <RouterProvider router={router} />
}

export default App
