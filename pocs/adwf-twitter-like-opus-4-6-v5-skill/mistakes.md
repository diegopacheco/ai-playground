# Mistakes Log

- Rust 2024 edition requires `unsafe` blocks for `std::env::set_var` and `std::env::remove_var`. Config tests using env var manipulation must wrap calls in `unsafe {}`.
- Parallel Rust tests that mutate env vars cause race conditions. Replaced env-mutating config tests with simpler struct/field tests.
- Vitest picks up Playwright e2e test files by default. Must add `exclude: ["e2e/**", "node_modules/**"]` to vitest config.
