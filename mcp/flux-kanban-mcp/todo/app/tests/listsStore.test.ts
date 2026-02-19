import { getLists, saveLists } from '../src/store/listsStore'
import type { GroceryList } from '../src/types'

beforeEach(() => {
  localStorage.clear()
})

test('getLists returns empty array when storage is empty', () => {
  expect(getLists()).toEqual([])
})

test('saveLists persists lists to localStorage', () => {
  const list: GroceryList = { id: '1', name: 'Shopping', items: [], createdAt: 1000 }
  saveLists([list])
  expect(localStorage.getItem('grocery_lists')).not.toBeNull()
})

test('getLists returns previously saved lists', () => {
  const list: GroceryList = { id: '1', name: 'Shopping', items: [], createdAt: 1000 }
  saveLists([list])
  expect(getLists()).toEqual([list])
})

test('getLists returns empty array on corrupted storage', () => {
  localStorage.setItem('grocery_lists', 'not valid json')
  expect(getLists()).toEqual([])
})

test('saveLists overwrites previous data', () => {
  const list1: GroceryList = { id: '1', name: 'First', items: [], createdAt: 1000 }
  const list2: GroceryList = { id: '2', name: 'Second', items: [], createdAt: 2000 }
  saveLists([list1])
  saveLists([list2])
  expect(getLists()).toEqual([list2])
})

test('saveLists preserves all list fields', () => {
  const list: GroceryList = {
    id: 'abc',
    name: 'My List',
    items: [{ id: 'i1', name: 'Milk', done: false, createdAt: 500 }],
    createdAt: 9999,
  }
  saveLists([list])
  expect(getLists()[0]).toEqual(list)
})

test('getLists handles multiple lists', () => {
  const lists: GroceryList[] = [
    { id: '1', name: 'Fruits', items: [], createdAt: 100 },
    { id: '2', name: 'Veggies', items: [], createdAt: 200 },
    { id: '3', name: 'Dairy', items: [], createdAt: 300 },
  ]
  saveLists(lists)
  expect(getLists()).toHaveLength(3)
  expect(getLists().map(l => l.name)).toEqual(['Fruits', 'Veggies', 'Dairy'])
})
