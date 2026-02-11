# Twitter Clone

A full-stack Twitter clone application with a Rust backend and React frontend.

## 📋 Project Overview

**Status**: Development Complete (75% Production Ready)
**Grade**: B+ - Solid implementation with good fundamentals
**Date**: 2026-02-09
**Test Coverage**: 106 tests (Integration, E2E, Performance)

### Quick Links

📚 **Documentation**
- [Design Document](design-doc.md) - Architecture and implementation details
- [Feature Documentation](review/2026-02-09/features.md) - Complete feature guide
- [API Documentation](API_DOCUMENTATION.md) - API reference
- [Changelog](changelog.md) - Complete project history

🔍 **Reviews**
- [Code Review](review/2026-02-09/code-review.md) - Quality assessment and recommendations
- [Security Review](review/2026-02-09/sec-review.md) - Security audit and vulnerabilities
- [Implementation Summary](review/2026-02-09/summary.md) - Comprehensive project summary

🚀 **Quick Start Guides**
- [Quick Start](QUICKSTART.md) - Get started in 5 minutes
- [Development Guide](DEVELOPMENT.md) - Development best practices
- [Deployment Guide](DEPLOYMENT.md) - Production deployment

### Summary Highlights

**What Was Built:**
- 21 REST API endpoints (Auth, Users, Tweets, Comments)
- React 19 frontend with TypeScript, Vite, Tailwind CSS
- PostgreSQL database with 6 normalized tables
- JWT authentication with bcrypt password hashing
- 77 source files (~20,120 lines of code)
- 106 comprehensive tests (13 integration, 86 e2e, 7 performance)

**Key Features:**
- User authentication (register, login, JWT)
- User profiles with followers/following
- Tweet creation, viewing, deletion (280 char limit)
- Social interactions (likes, retweets, comments)
- Feed generation from followed users
- Pagination and filtering

**Critical Issues to Fix Before Production:**
1. 🔴 **JWT Secret**: Configure proper secret (not default value)
2. 🔴 **CORS**: Restrict to specific allowed origins
3. 🟡 **Rate Limiting**: Add to all endpoints
4. 🟡 **N+1 Queries**: Optimize tweet/comment enrichment
5. 🟡 **Type Mismatch**: Fix LoginCredentials in frontend

**Security Posture:**
- 2 Critical vulnerabilities
- 4 High risk issues
- 5 Medium risk issues
- 3 Low risk issues
- See [Security Review](review/2026-02-09/sec-review.md) for details

---

## Backend - Rust with Axum

A high-performance Twitter clone backend built with Rust, Axum, and PostgreSQL.

## Features

- JWT-based authentication
- User registration and login
- Tweet creation, deletion, and feed
- Follow/unfollow users
- Like and retweet tweets
- Comment on tweets
- User profiles with statistics
- CORS support for frontend integration
- Connection pooling with SQLx
- Comprehensive error handling
- Database migrations

## Tech Stack

- Rust 1.93+ (Edition 2024)
- Axum web framework
- Tokio async runtime
- SQLx for PostgreSQL
- JWT for authentication
- bcrypt for password hashing
- Tracing for logging

## Prerequisites

- Rust 1.93 or later
- PostgreSQL 14 or later
- cargo

## Setup

1. Install PostgreSQL and create a database:
```bash
createdb twitter
```

2. Copy the environment template:
```bash
cp .env.template .env
```

3. Update the `.env` file with your database credentials and JWT secret.

4. Install dependencies and build:
```bash
cargo build --release
```

## Running the Server

```bash
cargo run --release
```

The server will start on `http://0.0.0.0:8000`.

## Database Migrations

Migrations run automatically on server startup. Manual migration:

```bash
cargo install sqlx-cli --no-default-features --features postgres
sqlx migrate run
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `POST /api/auth/logout` - Logout user

### Users
- `GET /api/users/:id` - Get user profile
- `PUT /api/users/:id` - Update user profile
- `GET /api/users/:id/followers` - Get user followers
- `GET /api/users/:id/following` - Get users being followed
- `POST /api/users/:id/follow` - Follow a user
- `DELETE /api/users/:id/follow` - Unfollow a user

### Tweets
- `POST /api/tweets` - Create new tweet
- `GET /api/tweets/:id` - Get single tweet
- `DELETE /api/tweets/:id` - Delete tweet
- `GET /api/tweets/feed` - Get user feed (tweets from followed users)
- `GET /api/tweets/user/:userId` - Get tweets from specific user
- `POST /api/tweets/:id/like` - Like a tweet
- `DELETE /api/tweets/:id/like` - Unlike a tweet
- `POST /api/tweets/:id/retweet` - Retweet a tweet
- `DELETE /api/tweets/:id/retweet` - Remove retweet

### Comments
- `POST /api/tweets/:id/comments` - Add comment to tweet
- `GET /api/tweets/:id/comments` - Get comments for tweet
- `DELETE /api/comments/:id` - Delete comment

## Environment Variables

- `DATABASE_URL` - PostgreSQL connection string
- `JWT_SECRET` - Secret key for JWT token generation
- `RUST_LOG` - Logging level configuration

## Project Structure

```
src/
├── main.rs           - Application entry point
├── config.rs         - Configuration management
├── state.rs          - Application state
├── error.rs          - Error handling
├── middleware/
│   ├── mod.rs
│   └── auth.rs       - JWT authentication middleware
├── models/
│   ├── mod.rs
│   ├── user.rs       - User models
│   ├── tweet.rs      - Tweet models
│   ├── follow.rs     - Follow models
│   ├── like.rs       - Like models
│   ├── retweet.rs    - Retweet models
│   └── comment.rs    - Comment models
├── handlers/
│   ├── mod.rs
│   ├── auth.rs       - Authentication handlers
│   ├── users.rs      - User handlers
│   ├── tweets.rs     - Tweet handlers
│   └── comments.rs   - Comment handlers
└── routes/
    └── mod.rs        - Route definitions

migrations/          - Database migrations
```

## Testing

Run Rust tests:
```bash
cargo test
```

Run integration tests:
```bash
./run-integration-tests.sh
```

Run E2E tests with Playwright:
```bash
./run-e2e-tests.sh
```

View Playwright test report:
```bash
npx playwright show-report
```

Run K6 performance tests:
```bash
./run-k6-tests.sh
```

## Development

Format code:
```bash
cargo fmt
```

Lint code:
```bash
cargo clippy
```

## License

MIT

---

## Frontend - React with TypeScript

A modern Twitter clone frontend built with React 19, TypeScript, Vite, and Tailwind CSS.

### Frontend Tech Stack

- React 19
- TypeScript
- Vite
- Bun
- TanStack Query (React Query)
- React Router DOM
- Tailwind CSS

### Frontend Prerequisites

- Bun installed on your system
- Backend API running on http://localhost:8000

### Frontend Installation

Install dependencies using Bun:

```bash
bun install
```

### Frontend Development

Start the development server:

```bash
bun run dev
```

The application will be available at http://localhost:5173

### Frontend Build

Build the application for production:

```bash
bun run build
```

### Frontend Features

- User authentication (login/register)
- Create, view, and delete tweets
- Like and retweet functionality
- Comment on tweets
- User profiles
- Follow/unfollow users
- View followers and following lists
- Responsive design with Tailwind CSS

### Frontend Project Structure

```
src/
├── components/       # Reusable React components
│   ├── CommentList.tsx
│   ├── FeedList.tsx
│   ├── NavigationBar.tsx
│   ├── TweetCard.tsx
│   ├── TweetComposer.tsx
│   └── UserCard.tsx
├── contexts/        # React Context providers
│   └── AuthContext.tsx
├── lib/            # Utility functions and API client
│   └── api.ts
├── pages/          # Page components
│   ├── HomePage.tsx
│   ├── LoginPage.tsx
│   ├── ProfilePage.tsx
│   └── TweetDetailPage.tsx
├── types/          # TypeScript type definitions
│   └── index.ts
├── App.tsx         # Main application component
├── main.tsx        # Application entry point
└── index.css       # Global styles
```

### API Configuration

The frontend is configured to communicate with the backend API at:

```
http://localhost:8000/api
```

To change this, modify the `API_BASE_URL` constant in `src/lib/api.ts`.

### Authentication

The application uses JWT token-based authentication. Tokens are stored in localStorage and automatically included in API requests via the Authorization header.
