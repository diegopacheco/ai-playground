# BTCA

https://btca.dev/

## Usage

```
btca init
btca add -n svelte-dev https://github.com/sveltejs/svelte.dev
```
I had to login with GTP subscription and pick up codex-5.3 model

```
btca ask -r svelte-dev -q "How do I define remote functions?"
```

```
loading resources...
creating collection...

In SvelteKit, you define remote functions in a **`.remote.js`** (or `.remote.ts`) module by exporting functions wrapped with one of these helpers from `@sveltejs/kit`:

- `query(...)` — read-only remote call
- `command(...)` — mutation/action remote call
- `prerender(...)` — prerender-time remote function

Basic pattern:

``js
// src/lib/todos.remote.js
import { query, command } from '@sveltejs/kit';

export const getTodos = query(async () => {
        // read data
        return [{ id: 1, text: 'Ship it' }];
});

export const addTodo = command(async (text) => {
        // mutate data
        return { ok: true };
});
``

You then import and call these from your app code (including components), and SvelteKit handles the client/server boundary for you.

Key requirements/caveats:
1. Put them in a `*.remote.js|ts` file.
2. Export wrapped functions (`query`, `command`, or `prerender`) rather than plain exports.
3. Use `command` for side effects/mutations and `query` for fetch/read semantics.

Sources:
- [documentation/docs/02-kit/02-core-concepts/06-remote-functions.md](https://github.com/sveltejs/svelte.dev/blob/main/documentation/docs/02-kit/02-core-concepts/06-remote-functions.md)
```