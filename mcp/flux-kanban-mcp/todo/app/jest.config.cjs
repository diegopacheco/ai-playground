/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: 'jsdom',
  testMatch: ['<rootDir>/tests/**/*.test.{ts,tsx}'],
  transform: {
    '^.+\\.(ts|tsx)$': ['ts-jest', { tsconfig: './tsconfig.test.json' }],
  },
  setupFilesAfterEnv: ['@testing-library/jest-dom'],
}
