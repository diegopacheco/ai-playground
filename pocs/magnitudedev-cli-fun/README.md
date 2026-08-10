# Magnitude - TanStack Table V9 Application

A clean, modular, light-themed data table application built with React, TypeScript, and Node.js, powered by TanStack Table V9.

## Model

```
Qwen 3.5 9B
```

## Prompt 

```
TanStack Table V9 its out I want a app with react , typescript 7x, nodejs, start/stop script and a tabler using TanStack Table V9. Lighthemed website, very modular and clean code. build.
```

## Instal

```
https://magnitude.dev/
```
```
npm install -g @magnitudedev/cli

```

## Features

- **TanStack Table V9**: Industry-standard table library with excellent performance
- **Light-themed UI**: Refined, minimalist design with sophisticated typography
- **Modular Architecture**: Clean separation of concerns with TypeScript
- **Start/Stop Scripts**: Easy server management
- **Responsive Design**: Works on desktop and mobile devices
- **Type-Safe**: Full TypeScript support for development and production

## Tech Stack

- **Frontend**: React 18, TypeScript 5
- **Table Library**: TanStack Table V9
- **Routing**: React Router DOM
- **Date Handling**: date-fns
- **Build Tool**: Vite
- **Backend**: Node.js static file server

## Project Structure

```
├── src/
│   ├── App.tsx              # Main application component
│   ├── main.tsx             # Entry point
│   └── styles.css           # Global styles
├── dist/
│   └── server/              # Node.js server scripts
├── start                    # Start script
├── stop                     # Stop script
└── package.json
```

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
npm install
```

### Development

```bash
# Build first
npm run build

# Start the server (Vite preview)
npm run preview
```

The application will open at http://localhost:4173 (Vite's default port)

### Production

```bash
npm run build
npm run preview
```

### Server Management

```bash
# Start the server
./start

# Stop the server
./stop
```

## Features

### Dashboard
- Overview statistics with key metrics
- Recent activity feed
- Quick access to main sections

### Users Table
- Full-featured data table with TanStack Table V9
- Sorting capabilities
- Pagination
- Search functionality
- Status badges (Active, Inactive, Pending)
- Role badges with color coding
- Edit and delete actions
- Responsive design

### Analytics (Coming Soon)
- Data visualization
- Charts and graphs
- Performance metrics

### Settings (Coming Soon)
- Application configuration
- User preferences
- System settings

## Design Principles

- **Refined Minimalism**: Clean lines, generous whitespace
- **Sophisticated Typography**: System fonts with careful weight selection
- **Subtle Depth**: Light shadows and borders for visual hierarchy
- **Professional Aesthetic**: Enterprise-ready design
- **Accessibility**: Semantic HTML, proper contrast ratios

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

MIT
