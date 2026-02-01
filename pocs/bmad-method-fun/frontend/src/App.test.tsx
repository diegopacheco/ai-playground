import { render, screen, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import App from './App'

describe('App', () => {
  it('starts in start screen with idle status', () => {
    render(<App />)
    expect(screen.getByRole('button', { name: /start game/i })).toBeInTheDocument()
    expect(screen.getByText(/status:\s*idle/i)).toBeInTheDocument()
    expect(screen.queryByTestId('board')).not.toBeInTheDocument()
  })

  it('starts a new game and sets status to running', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    expect(screen.getByText(/status:\s*running/i)).toBeInTheDocument()
  })

  it('renders board and initial piece when running', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    expect(screen.getByTestId('board')).toBeInTheDocument()
    expect(screen.getByTestId('piece')).toBeInTheDocument()
  })

  it('forces a drop and resolves placement at bottom', () => {
    vi.useFakeTimers()
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    const initialPiece = screen.getByTestId('piece')
    const initialKey = initialPiece.getAttribute('data-testid')
    vi.advanceTimersByTime(40000)
    const movedPiece = screen.getByTestId('piece')
    expect(movedPiece).toBeInTheDocument()
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    expect(initialKey).toBe('piece')
  })
})
