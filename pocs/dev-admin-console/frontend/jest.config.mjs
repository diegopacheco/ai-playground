export default {
  testEnvironment: "jsdom",
  setupFilesAfterEnv: ["<rootDir>/src/test/setup.ts"],
  transform: {
    "^.+\\.(t|j)sx?$": ["@swc/jest", {
      jsc: {
        parser: { syntax: "typescript", tsx: true },
        transform: { react: { runtime: "automatic" } }
      }
    }]
  },
  moduleNameMapper: {
    "\\.(css|less|scss)$": "<rootDir>/src/test/styleMock.ts",
    "^@design/(.*)$": "<rootDir>/src/design-system/$1",
    "^@console/(.*)$": "<rootDir>/src/console/$1",
    "^@engines/(.*)$": "<rootDir>/src/engines/$1",
    "^@lib/(.*)$": "<rootDir>/src/lib/$1"
  },
  testMatch: ["<rootDir>/src/**/*.test.{ts,tsx}"],
  collectCoverageFrom: ["src/**/*.{ts,tsx}", "!src/**/*.stories.tsx", "!src/test/**"]
};
