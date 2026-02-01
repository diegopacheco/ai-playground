import { render, screen, fireEvent, act } from '@testing-library/react'
import { vi } from 'vitest'
import App from './App'

const rows = 20
const forcedDropMs = 40000

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

  it('increments score on forced drop placement', () => {
    vi.useFakeTimers()
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    act(() => {
      vi.advanceTimersByTime(forcedDropMs * (rows - 1))
    })
    expect(screen.getByText(/score:\s*10/i)).toBeInTheDocument()
    vi.useRealTimers()
  })

  it('increments level when score reaches 100', () => {
    vi.useFakeTimers()
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    act(() => {
      vi.advanceTimersByTime(forcedDropMs * (rows - 1) * 10)
    })
    expect(screen.getByText(/level:\s*2/i)).toBeInTheDocument()
    vi.useRealTimers()
  })
})
