import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from '@tanstack/react-router'
import { router } from '../App'

describe('HomePage', () => {
  it('renders hero heading', async () => {
    const qc = new QueryClient()
    render(
      <QueryClientProvider client={qc}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    )
    const heading = await screen.findByText('Retirement')
    expect(heading).toBeInTheDocument()
  })
})
