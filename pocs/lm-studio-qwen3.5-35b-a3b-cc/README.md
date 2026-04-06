## Model

```
Qwen 3.5, 35B A3B
```

## Prompt

```
Build a paper, rock, cissors game in Typescript. You muse use vite, bun, react 19 and TanStack, make sure you update my readme at the and, dont delete waht I already have and make sure you create a run.sh and stop.sh. You also will create 2 pages - one to play the game and other to show all historical games results.
```

## Experience Notes

* LM Studio 0.49
* IT loose my read me - I had to copy it back.
* Qwen 3.5 was faster than GLM 4.7 flash to produce code.
* You can see tool usage but it does not tell to you want he is doing.
* Qwen use the right version of vite 8x and react 19.
* Code was generate relatively fast how ever the page was empty/blank so it could not one shot it.
* Qwen 3.5 used my playwright mcp to check what was wrong with the app. Smart move.
* Time to time this model stop and was doing nothing - I had to poke it all the time
* 

# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:
```

## Rock Paper Scissors Game

A simple rock paper scissors game built with React 19 and TanStack Router.

### Features

* Play rock paper scissors against the computer
* View all historical results
* Statistics tracking (wins, losses, draws)
* Results persist in localStorage

### Scripts

```bash
./run.sh    # Start development server
./stop.sh   # Stop development server
```

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
