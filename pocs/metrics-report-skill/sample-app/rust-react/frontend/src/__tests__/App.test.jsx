import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from '@tanstack/react-router'
import { router } from '../App'

function renderApp() {
  const qc = new QueryClient()
  render(
    <QueryClientProvider client={qc}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  )
}

describe('App', () => {
  it('renders the nav logo', async () => {
    renderApp()
    const logos = await screen.findAllByText('RetireSmart')
    expect(logos.length).toBeGreaterThan(0)
  })
})
