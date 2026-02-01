import { render, screen, fireEvent } from '@testing-library/react'
import App from './App'

describe('App', () => {
  it('starts in start screen with idle status', () => {
    render(<App />)
    expect(screen.getByRole('button', { name: /start game/i })).toBeInTheDocument()
    expect(screen.getByText(/status:\s*idle/i)).toBeInTheDocument()
  })

  it('starts a new game and sets status to running', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /start game/i }))
    expect(screen.getByText(/status:\s*running/i)).toBeInTheDocument()
  })
})
