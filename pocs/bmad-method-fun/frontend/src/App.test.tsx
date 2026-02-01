import { render, screen, fireEvent, act } from '@testing-library/react'
import { vi } from 'vitest'
import App from './App'

const rows = 20
const forcedDropMs = 40000
const maxPlacements = 10

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
    expect(screen.getByText(/game over/i)).toBeInTheDocument()
    expect(screen.getByText(/final score:\s*100/i)).toBeInTheDocument()
    vi.useRealTimers()
  })

  it('shows forced drop and board expansion timers', () => {
    vi.useFakeTimers()
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    expect(screen.getByText(/forced drop in:/i)).toBeInTheDocument()
    expect(screen.getByText(/board expands in:/i)).toBeInTheDocument()
    vi.useRealTimers()
  })

  it('ends the session after max placements', () => {
    vi.useFakeTimers()
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    act(() => {
      vi.advanceTimersByTime(forcedDropMs * (rows - 1) * maxPlacements)
    })
    expect(screen.getByText(/status:\s*ended/i)).toBeInTheDocument()
    expect(screen.getByText(/game over/i)).toBeInTheDocument()
    vi.useRealTimers()
  })

  it('opens admin configuration interface', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /admin settings/i }))
    expect(screen.getByRole('region', { name: /admin configuration/i })).toBeInTheDocument()
    expect(screen.getByText(/level backgrounds/i)).toBeInTheDocument()
    expect(screen.getByText(/timers/i)).toBeInTheDocument()
  })
})
