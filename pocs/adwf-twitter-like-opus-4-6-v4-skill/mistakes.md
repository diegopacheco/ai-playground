# Mistakes Log

- Bash permission may be denied during agent execution; always verify scripts can be chmod'd and run before marking task complete.
- Model structs (Like, Follow) not directly constructed in handlers cause dead_code warnings; add #[allow(dead_code)] to suppress.
- AuthUser fields read via the extractor pattern are not detected by the compiler as "read"; add #[allow(dead_code)] to the struct.
- Use runtime sqlx::query / sqlx::query_as (not the macro variants) to avoid needing DATABASE_URL at compile time for SQLite.
- When using Vite with `tsc -b`, must split tsconfig into tsconfig.json (references), tsconfig.app.json (app code), and tsconfig.node.json (config files); single tsconfig with `include: ["src"]` breaks `tsc -b`.
- With `noUnusedParameters: true`, avoid underscore-prefixed unused params; instead restructure the code to actually use all parameters or remove them.
- Test files (*.test.tsx) must be excluded from tsconfig.app.json to avoid build failures from test-only imports (vitest, @testing-library).
- With `verbatimModuleSyntax: true`, always use `import type` for type-only imports; mixing value and type imports causes errors.
- vitest.config.ts must NOT be included in tsconfig.node.json to avoid Plugin type conflicts between vitest's bundled vite and the project's vite.
