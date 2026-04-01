import '@testing-library/jest-dom'

global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
}
