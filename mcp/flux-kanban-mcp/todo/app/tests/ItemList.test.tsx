import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ItemList } from '../src/components/ItemList'
import type { GroceryItem } from '../src/types'

function Wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>
}

const items: GroceryItem[] = [
  { id: '1', name: 'Milk', done: false, createdAt: 1000 },
  { id: '2', name: 'Eggs', done: true, createdAt: 2000 },
  { id: '3', name: 'Bread', done: false, createdAt: 3000 },
]

test('renders all item names', () => {
  render(<Wrapper><ItemList listId="list1" items={items} /></Wrapper>)
  expect(screen.getByText('Milk')).toBeInTheDocument()
  expect(screen.getByText('Eggs')).toBeInTheDocument()
  expect(screen.getByText('Bread')).toBeInTheDocument()
})

test('shows empty state when no items', () => {
  render(<Wrapper><ItemList listId="list1" items={[]} /></Wrapper>)
  expect(screen.getByText(/No items/)).toBeInTheDocument()
})

test('done items have checked checkbox', () => {
  render(<Wrapper><ItemList listId="list1" items={items} /></Wrapper>)
  const checkboxes = screen.getAllByRole('checkbox') as HTMLInputElement[]
  expect(checkboxes[0].checked).toBe(false)
  expect(checkboxes[1].checked).toBe(true)
  expect(checkboxes[2].checked).toBe(false)
})

test('renders correct number of items', () => {
  render(<Wrapper><ItemList listId="list1" items={items} /></Wrapper>)
  expect(screen.getAllByRole('checkbox')).toHaveLength(3)
})

test('done item has item-done class', () => {
  const { container } = render(<Wrapper><ItemList listId="list1" items={items} /></Wrapper>)
  const doneItems = container.querySelectorAll('.item-done')
  expect(doneItems).toHaveLength(1)
})
