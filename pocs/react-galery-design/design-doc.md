# Prompt Maker - Design Document

## Overview

Prompt Maker is a React application inspired by Figma Make, but entirely prompt-driven with no visual drag and drop. The app has a tab-style wizard with 3 steps where users describe what frontend app they want, watch it being built, and preview the generated HTML/CSS output.

## Technology Stack

### Frontend
- React 19
- Bun (runtime/package manager)
- Vite 6 (build tool)
- TypeScript 5.x
- TanStack Router (wizard step routing)
- TanStack Form (prompt form handling)
- TanStack Store (global wizard state)
- CSS Modules (scoped styling)

All TypeScript. No external UI or CSS libraries.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Frontend (React 19)                       │
├──────────────────────────────────────────────────────────────┤
│  PromptStep  │  MakingStep (Progress)  │  PreviewStep        │
└──────────────┴─────────────────────────┴─────────────────────┘
                            │
                            │ TanStack Store
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     Generation Engine                         │
├──────────────────────────────────────────────────────────────┤
│  Parser  │  Templates  │  Fragments  │  Style Modules        │
└──────────┴─────────────┴─────────────┴───────────────────────┘
```

## UI Screens

### 1. Prompt Step (Tab 1)

**Purpose**: User describes the frontend app they want to build.

**Components**:
- Text area: Large input (6-8 rows) where user types a free-form prompt describing the app
- Style dropdown: `<select>` with 13 UX style options
- Generate button: Starts the generation process

**UX Style Options**:

| # | Style | Description |
|---|-------|-------------|
| 1 | Terminal | Dark background, monospace font, green/amber text, CLI aesthetic |
| 2 | Modern | Clean lines, rounded corners, gradients, sans-serif, whitespace-heavy |
| 3 | Traditional | Classic layout, serif fonts, muted colors, newspaper-like structure |
| 4 | Brutalist | Raw, bold typography, high contrast, intentionally rough edges |
| 5 | Glassmorphism | Frosted glass effects, transparency, blur backgrounds, soft borders |
| 6 | Neomorphism | Soft shadows, extruded look, muted pastels, embossed elements |
| 7 | Retro | Pixel fonts, neon colors, 80s/90s vibe, grid patterns |
| 8 | Minimalist | Ultra-clean, lots of whitespace, black and white, subtle accents |
| 9 | Corporate | Professional blue tones, structured grid, standard UI patterns |
| 10 | Playful | Bright colors, rounded shapes, fun animations, handwritten fonts |
| 11 | Dark Mode | Dark backgrounds, light text, accent colors, reduced eye strain |
| 12 | Material | Material Design inspired, elevation shadows, bold colors, ripple feel |
| 13 | Flat | No shadows, no gradients, solid colors, crisp edges, simple shapes |

**Validation**:
- Generate button disabled if text area is empty

### 2. Making Step (Tab 2)

**Purpose**: Shows progress while the app is being assembled.

**Components**:
- Progress bar: Horizontal, fills from 0% to 100%, animated
- Status text: Shows what is currently happening below the bar

**Status Messages (shown sequentially)**:
- 0-10%: "Parsing your prompt..."
- 10-25%: "Selecting layout structure..."
- 25-40%: "Building components..."
- 40-60%: "Applying [style name] theme..."
- 60-75%: "Generating CSS..."
- 75-90%: "Assembling HTML..."
- 90-100%: "Finalizing your app..."

Progress is simulated over ~3 seconds using `setInterval`. The actual generation is near-instant. When progress reaches 100%, auto-transitions to Step 3 after 500ms.

### 3. Preview Step (Tab 3)

**Purpose**: Display the generated frontend app.

**Components**:
- Preview/Code toggle: Switch between visual preview and raw code
- Iframe: Full-width preview rendering the generated HTML/CSS
- Code viewer: Read-only display of raw HTML/CSS with basic syntax highlighting (CSS-based, no library)
- Copy HTML button: Copies source to clipboard
- Download button: Downloads as `index.html` with CSS inlined in a `<style>` tag, also saves to `solutions/` folder
- Start Over button: Resets everything, returns to Step 1

## Tab Navigation

Tabs are always visible at the top: "1. Prompt" | "2. Making" | "3. Preview"

- Active tab: bold text, indigo bottom border (3px)
- Completed tabs: clickable to go back
- Disabled tabs (not yet reached): dimmed, not clickable
- Each tab maps to a TanStack Router route

## Project Structure

```
react-galery-design/
├── design-doc.md
├── run.sh
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   └── src/
└── solutions/
    └── (generated HTML files saved here by the app)
```

## Frontend Structure

```
frontend/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── index.html
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── App.module.css
│   ├── routeTree.gen.ts
│   ├── routes/
│   │   ├── __root.tsx
│   │   ├── prompt.tsx
│   │   ├── making.tsx
│   │   └── preview.tsx
│   ├── components/
│   │   ├── TabBar.tsx
│   │   ├── TabBar.module.css
│   │   ├── ProgressBar.tsx
│   │   ├── ProgressBar.module.css
│   │   ├── CodeViewer.tsx
│   │   └── CodeViewer.module.css
│   ├── store/
│   │   └── wizard.ts
│   ├── engine/
│   │   ├── parser.ts
│   │   ├── generator.ts
│   │   ├── styles/
│   │   │   ├── terminal.ts
│   │   │   ├── modern.ts
│   │   │   ├── traditional.ts
│   │   │   ├── brutalist.ts
│   │   │   ├── glassmorphism.ts
│   │   │   ├── neomorphism.ts
│   │   │   ├── retro.ts
│   │   │   ├── minimalist.ts
│   │   │   ├── corporate.ts
│   │   │   ├── playful.ts
│   │   │   ├── darkmode.ts
│   │   │   ├── material.ts
│   │   │   └── flat.ts
│   │   ├── fragments/
│   │   │   ├── navbar.ts
│   │   │   ├── hero.ts
│   │   │   ├── cardGrid.ts
│   │   │   ├── footer.ts
│   │   │   ├── features.ts
│   │   │   ├── contactForm.ts
│   │   │   ├── sidebar.ts
│   │   │   ├── table.ts
│   │   │   ├── gallery.ts
│   │   │   ├── stats.ts
│   │   │   ├── pricing.ts
│   │   │   └── cta.ts
│   │   └── css/
│   │       ├── base.ts
│   │       └── responsive.ts
│   └── types/
│       └── index.ts
```

## Types

```typescript
type UxStyle =
  | "terminal"
  | "modern"
  | "traditional"
  | "brutalist"
  | "glassmorphism"
  | "neomorphism"
  | "retro"
  | "minimalist"
  | "corporate"
  | "playful"
  | "darkmode"
  | "material"
  | "flat";

type WizardStep = 1 | 2 | 3;

type WizardState = {
  currentStep: WizardStep;
  prompt: string;
  selectedStyle: UxStyle;
  progress: number;
  statusMessage: string;
  generatedHtml: string;
  isGenerating: boolean;
};
```

## TanStack Usage

### TanStack Store (wizard state)

```typescript
import { Store } from '@tanstack/store';

const wizardStore = new Store<WizardState>({
  currentStep: 1,
  prompt: "",
  selectedStyle: "modern",
  progress: 0,
  statusMessage: "",
  generatedHtml: "",
  isGenerating: false,
});
```

Components use `useStore(wizardStore, (s) => s.field)` to subscribe to slices.

### TanStack Router (step navigation)

| Route | Step | Component |
|-------|------|-----------|
| `/` or `/prompt` | Step 1 | PromptStep |
| `/making` | Step 2 | MakingStep |
| `/preview` | Step 3 | PreviewStep |

Route guards:
- `/making` redirects to `/prompt` if prompt is empty
- `/preview` redirects to `/making` if generatedHtml is empty

### TanStack Form (prompt input)

```typescript
const form = useForm({
  defaultValues: {
    prompt: "",
    style: "modern" as UxStyle,
  },
  onSubmit: ({ value }) => {
    wizardStore.setState((prev) => ({
      ...prev,
      prompt: value.prompt,
      selectedStyle: value.style,
    }));
    startGeneration();
    navigate({ to: "/making" });
  },
});
```

## Prompt Parser

The parser scans the prompt for keywords and maps them to HTML fragments.

| Keywords | Fragment |
|----------|---------|
| nav, navigation, menu, header | navbar |
| hero, banner, headline, welcome | hero |
| card, grid, tiles, items | cardGrid |
| footer, bottom, copyright | footer |
| feature, benefit, advantage | features |
| contact, form, email, message | contactForm |
| sidebar, aside, menu | sidebar |
| table, data, list, rows | table |
| gallery, images, photos, portfolio | gallery |
| stats, metrics, numbers, counter | stats |
| pricing, plan, tier, subscription | pricing |
| cta, call to action, sign up, get started | cta |

Default (no keywords match): navbar + hero + cardGrid + footer.

## Generation Engine

1. Parse prompt to determine which fragments to include
2. Get style module for the selected UxStyle
3. Build HTML: DOCTYPE + head + `<style>` tag + body with fragments in order
4. Each fragment function receives style config and returns HTML string
5. Each style module exports CSS variables and component-specific overrides
6. Combine base CSS + style CSS + responsive CSS into a single `<style>` block
7. Output: one complete `index.html` string

## UI Design

- Background: #f8f9fa
- Container: white, max-width 1000px, centered, rounded corners, subtle shadow
- Primary accent: #4f46e5 (indigo)
- Typography: system font stack
- Spacing: 8px base unit
- Tab active: indigo bottom border (3px), bold
- Tab inactive: gray text, dimmed
- Generate button: full-width, indigo bg, white text, 48px height
- Progress bar: 12px height, rounded, indigo fill, animated
- Iframe preview: full-width, 500px min-height, 1px solid border

## Responsive Behavior

| Breakpoint | Behavior |
|------------|----------|
| >= 768px | Full layout, side padding |
| < 768px | Full-width, reduced padding, stacked buttons |

## Implementation Details

### Claude Code Execution Strategy

The implementation will be done using Claude Code with `-p` flag in YOLO mode (no interactive questions). Each implementation step runs as a single Claude invocation.

For the UI generation in Step 2 (Making), the `frontend-design` skill is used to deliver high-quality, production-grade HTML/CSS output. The frontend-design skill ensures the generated output avoids generic AI aesthetics and produces distinctive, polished designs.

### Solutions Folder

All generated HTML/CSS apps are saved into the `solutions/` folder at the project root. Each generated file is named with a timestamp: `solution-{timestamp}.html`. The solutions folder is created automatically if it does not exist.

### run.sh

A `run.sh` script at the project root installs dependencies and starts the dev server:

```bash
#!/bin/bash
cd frontend && bun install && bun dev
```

### Build Commands

```bash
bun create vite frontend --template react-ts
cd frontend
bun install
bun add @tanstack/react-router @tanstack/react-form @tanstack/store
bun dev
```

## Implementation Plan

1. Scaffold Vite + React + TypeScript project with Bun
2. Install TanStack dependencies (Router, Form, Store)
3. Define types in `types/index.ts`
4. Set up TanStack Router with routes for each step
5. Build TanStack Store for wizard state
6. Build TabBar component
7. Build PromptStep with TanStack Form (text area + style dropdown + generate button)
8. Build ProgressBar component
9. Build MakingStep with progress simulation and status messages
10. Build prompt parser (keyword -> fragments mapping)
11. Build all 13 style modules (CSS for each UxStyle)
12. Build all HTML fragment generators
13. Build generation engine orchestrator
14. Build CodeViewer component
15. Build PreviewStep with iframe, code toggle, and action buttons
16. Wire everything in root route layout
17. Polish responsive layout and transitions
