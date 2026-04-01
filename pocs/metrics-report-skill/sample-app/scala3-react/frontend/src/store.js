let _results = null
let _inputData = null
let _listeners = []

export const store = {
  getResults: () => _results,
  getInputData: () => _inputData,
  setResults: (r) => { _results = r; _listeners.forEach(fn => fn()) },
  setInputData: (d) => { _inputData = d; _listeners.forEach(fn => fn()) },
  subscribe: (fn) => { _listeners.push(fn); return () => { _listeners = _listeners.filter(l => l !== fn) } },
}
