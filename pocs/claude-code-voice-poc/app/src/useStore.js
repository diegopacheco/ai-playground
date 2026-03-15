import { useState, useEffect } from 'react'
import { getState, subscribe } from './store.js'

export function useStore() {
  const [state, setState] = useState(getState())
  useEffect(() => subscribe(setState), [])
  return state
}
