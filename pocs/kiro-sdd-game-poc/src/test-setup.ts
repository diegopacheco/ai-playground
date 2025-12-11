import '@testing-library/jest-dom';
import { beforeEach } from 'bun:test';

// Setup DOM environment for Bun tests
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!DOCTYPE html><html><body><div id="root"></div></body></html>', {
  url: 'http://localhost',
  pretendToBeVisual: true,
  resources: 'usable'
});

global.window = dom.window as any;
global.document = dom.window.document;
global.navigator = dom.window.navigator;

// Mock localStorage for tests
const localStorageMock = {
  getItem: (key: string) => {
    return localStorageMock.store[key] || null;
  },
  setItem: (key: string, value: string) => {
    localStorageMock.store[key] = value;
  },
  removeItem: (key: string) => {
    delete localStorageMock.store[key];
  },
  clear: () => {
    localStorageMock.store = {};
  },
  store: {} as Record<string, string>
};

Object.defineProperty(global.window, 'localStorage', {
  value: localStorageMock
});

// Reset localStorage before each test
beforeEach(() => {
  localStorageMock.clear();
  // Clear the DOM body for each test
  document.body.innerHTML = '<div id="root"></div>';
});