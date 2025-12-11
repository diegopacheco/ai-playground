# Rock Paper Scissors Game

A web-based rock paper scissors game built with React 19 and Bun that provides an interactive gaming experience with animations and persistent score tracking.

## Features

- Interactive rock, paper, scissors gameplay
- Animated game states and transitions
- Persistent score tracking using localStorage
- Clean, responsive design
- Property-based testing with fast-check

## Technology Stack

- **Frontend**: React 19 with TypeScript
- **Runtime**: Bun
- **Testing**: Bun test runner + React Testing Library + fast-check
- **Styling**: CSS with animations

## Getting Started

### Prerequisites

- [Bun](https://bun.sh) runtime installed

### Installation

```bash
bun install
```

### Development

Start the development server:

```bash
bun run dev
```

Or use the convenient run script:

```bash
./run.sh
```

The game will be available at `http://localhost:3000`

### Building

Build for production:

```bash
bun run build
```

### Testing

Run all tests:

```bash
bun test
```

Run tests in watch mode:

```bash
bun run test:watch
```

## Project Structure

```
src/
├── components/     # React components
├── services/       # Business logic and services
├── styles/         # CSS files
├── tests/          # Test files
│   └── properties/ # Property-based tests
├── types/          # TypeScript type definitions
├── utils/          # Utility functions
└── App.tsx         # Main application component
```

## Game Rules

- Rock beats Scissors
- Scissors beats Paper
- Paper beats Rock
- Same choices result in a tie

Scores are automatically saved to your browser's local storage and persist between sessions.
