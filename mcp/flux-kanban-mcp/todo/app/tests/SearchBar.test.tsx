import { render, screen, fireEvent } from '@testing-library/react'
import { SearchBar } from '../src/components/SearchBar'

test('renders search input with placeholder', () => {
  render(<SearchBar value="" onChange={() => {}} />)
  expect(screen.getByPlaceholderText('Search items...')).toBeInTheDocument()
})

test('displays current value', () => {
  render(<SearchBar value="milk" onChange={() => {}} />)
  expect(screen.getByDisplayValue('milk')).toBeInTheDocument()
})

test('calls onChange with new value when user types', () => {
  const onChange = jest.fn()
  render(<SearchBar value="" onChange={onChange} />)
  fireEvent.change(screen.getByPlaceholderText('Search items...'), {
    target: { value: 'eggs' },
  })
  expect(onChange).toHaveBeenCalledWith('eggs')
})

test('renders an input of type search', () => {
  render(<SearchBar value="" onChange={() => {}} />)
  expect(screen.getByRole('searchbox')).toBeInTheDocument()
})
