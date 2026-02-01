import { render, screen, fireEvent, act } from '@testing-library/react'
import { vi } from 'vitest'
import App from './App'

const rows = 20
const forcedDropMs = 40000
const maxPlacementsPerLevel = 10

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
    expect(screen.getByText(/score:\s*100/i)).toBeInTheDocument()
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
      vi.advanceTimersByTime(forcedDropMs * (rows - 1) * maxPlacementsPerLevel * 10)
    })
    expect(screen.getByText(/status:\s*ended/i)).toBeInTheDocument()
    expect(screen.getByText(/game over/i)).toBeInTheDocument()
    vi.useRealTimers()
  })

  it('opens admin configuration interface', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /admin settings/i }))
    expect(screen.getByRole('region', { name: /admin configuration/i })).toBeInTheDocument()
    expect(screen.getByText(/background/i)).toBeInTheDocument()
    expect(screen.getByText(/difficulty/i)).toBeInTheDocument()
  })

  it('applies selected background to board', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    fireEvent.click(screen.getByRole('button', { name: /admin settings/i }))
    fireEvent.change(screen.getByRole('combobox', { name: /background/i }), { target: { value: 'sunset' } })
    expect(screen.getByTestId('board')).toHaveClass('sunset')
  })

  it('updates timer settings from admin interface', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    fireEvent.click(screen.getByRole('button', { name: /admin settings/i }))
    fireEvent.change(screen.getByLabelText(/forced drop seconds/i), { target: { value: '5' } })
    fireEvent.change(screen.getByLabelText(/board expand seconds/i), { target: { value: '7' } })
    expect(screen.getByText(/forced drop in:\s*5s/i)).toBeInTheDocument()
    expect(screen.getByText(/board expands in:\s*7s/i)).toBeInTheDocument()
  })

  it('updates difficulty and affects forced drop display', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    fireEvent.click(screen.getByRole('button', { name: /admin settings/i }))
    fireEvent.change(screen.getByRole('combobox', { name: /difficulty/i }), { target: { value: 'hard' } })
    expect(screen.getByText(/forced drop in:\s*32s/i)).toBeInTheDocument()
  })

  it('applies live configuration updates without restarting the session', () => {
    vi.useFakeTimers()
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    act(() => {
      vi.advanceTimersByTime(forcedDropMs * (rows - 1))
    })
    expect(screen.getByText(/score:\s*10/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /admin settings/i }))
    fireEvent.change(screen.getByRole('combobox', { name: /background/i }), { target: { value: 'matrix' } })
    fireEvent.change(screen.getByRole('combobox', { name: /difficulty/i }), { target: { value: 'hard' } })
    fireEvent.change(screen.getByLabelText(/forced drop seconds/i), { target: { value: '5' } })
    fireEvent.change(screen.getByLabelText(/board expand seconds/i), { target: { value: '8' } })
    expect(screen.getByText(/status:\s*running/i)).toBeInTheDocument()
    expect(screen.getByText(/score:\s*10/i)).toBeInTheDocument()
    expect(screen.getByText(/forced drop in:\s*4s/i)).toBeInTheDocument()
    expect(screen.getByText(/board expands in:\s*8s/i)).toBeInTheDocument()
    expect(screen.getByTestId('board')).toHaveClass('matrix')
    expect(screen.getByText(/placements:\s*1\//i)).toBeInTheDocument()
    vi.useRealTimers()
  })

  it('updates max levels from admin interface', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    fireEvent.click(screen.getByRole('button', { name: /admin settings/i }))
    fireEvent.change(screen.getByLabelText(/number of levels/i), { target: { value: '3' } })
    expect(screen.getByText(/placements:\s*0\/30/i)).toBeInTheDocument()
  })

  it('ends immediately when max levels lowered below placements', () => {
    vi.useFakeTimers()
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    act(() => {
      vi.advanceTimersByTime(forcedDropMs * (rows - 1) * 20)
    })
    fireEvent.click(screen.getByRole('button', { name: /admin settings/i }))
    fireEvent.change(screen.getByLabelText(/number of levels/i), { target: { value: '1' } })
    expect(screen.getByText(/status:\s*ended/i)).toBeInTheDocument()
    vi.useRealTimers()
  })

  it('caps level at the configured max levels', () => {
    vi.useFakeTimers()
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    fireEvent.click(screen.getByRole('button', { name: /admin settings/i }))
    fireEvent.change(screen.getByLabelText(/number of levels/i), { target: { value: '1' } })
    act(() => {
      vi.advanceTimersByTime(forcedDropMs * (rows - 1) * 5)
    })
    expect(screen.getByText(/level:\s*1/i)).toBeInTheDocument()
    vi.useRealTimers()
  })
})
