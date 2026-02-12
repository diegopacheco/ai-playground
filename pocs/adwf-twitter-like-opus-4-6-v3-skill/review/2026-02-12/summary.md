# Changes Summary - 2026-02-12

## What Was Built

A full-stack Twitter-like web application with the following capabilities:
- User registration and login with JWT authentication
- Create, view, and delete posts (max 280 characters)
- Like and unlike posts with optimistic UI updates
- Follow and unfollow users
- Timeline feed showing posts from followed users and own posts
- User profiles with follower/following counts
- Single post detail view

## Architecture

### Backend (Rust + Axum)
- **Framework**: Axum 0.8 with Tokio async runtime
- **Database**: SQLite via SQLx 0.8 with runtime-tokio feature
- **Authentication**: JWT tokens (jsonwebtoken 9) with 24h expiry
- **Password Hashing**: Argon2 0.5 with random salt per user
- **CORS**: tower-http 0.6 configured for frontend origin
- **Logging**: tracing + tracing-subscriber

### Frontend (React + TypeScript)
- **Framework**: React 19 with TypeScript via Vite
- **Routing**: TanStack Router with auth-guarded routes
- **Data Fetching**: TanStack Query with optimistic updates
- **Styling**: Tailwind CSS
- **Package Manager**: Bun

### Database (SQLite)
- 4 tables: users, posts, likes, follows
- Composite primary keys on likes and follows
- Indexes on frequently queried columns
- Foreign key constraints defined (not enforced at runtime - see review)

## Files Created

### Backend
| File | Purpose |
|---|---|
| `backend/Cargo.toml` | Rust project manifest with dependencies |
| `backend/src/main.rs` | Entry point, route definitions, unit tests |
| `backend/src/lib.rs` | Shared app builder and test pool for integration tests |
| `backend/src/auth.rs` | JWT creation, verification, auth middleware |
| `backend/src/db.rs` | Database pool creation and table initialization |
| `backend/src/errors.rs` | AppError enum with HTTP status mapping |
| `backend/src/models.rs` | Data structures for users, posts, likes, follows, requests, responses |
| `backend/src/routes/mod.rs` | Route module declarations |
| `backend/src/routes/auth.rs` | Register, login, me handlers |
| `backend/src/routes/posts.rs` | Create, get, delete, like, unlike, timeline, user posts handlers |
| `backend/src/routes/users.rs` | Get user, followers, following, follow, unfollow handlers |

### Frontend
| File | Purpose |
|---|---|
| `frontend/src/main.tsx` | React entry point with QueryClient and RouterProvider |
| `frontend/src/auth.ts` | Token and user storage utilities (localStorage) |
| `frontend/src/api.ts` | API client with fetch wrapper and auth header injection |
| `frontend/src/types.ts` | TypeScript interfaces for API responses |
| `frontend/src/router.tsx` | TanStack Router configuration with auth guard |
| `frontend/src/hooks/useAuth.ts` | Login, register, logout hooks |
| `frontend/src/hooks/usePosts.ts` | Timeline, post, create post, toggle like hooks |
| `frontend/src/hooks/useUsers.ts` | Profile, followers, following, toggle follow hooks |
| `frontend/src/pages/HomePage.tsx` | Home page with composer and timeline |
| `frontend/src/pages/LoginPage.tsx` | Login form page |
| `frontend/src/pages/RegisterPage.tsx` | Registration form page |
| `frontend/src/pages/ProfilePage.tsx` | User profile page |
| `frontend/src/pages/PostDetailPage.tsx` | Single post detail page |
| `frontend/src/components/Navbar.tsx` | Top navigation bar |
| `frontend/src/components/PostCard.tsx` | Post display component with like button |
| `frontend/src/components/PostComposer.tsx` | Post creation form with character counter |
| `frontend/src/components/Timeline.tsx` | Post list with loading and empty states |
| `frontend/src/components/FollowButton.tsx` | Follow/unfollow toggle button |

### Database
| File | Purpose |
|---|---|
| `db/schema.sql` | SQLite schema with tables, constraints, indexes |

### Tests
| File | Purpose |
|---|---|
| `backend/tests/integration_test.rs` | 20 integration tests covering all API endpoints |
| `frontend/tests/app.spec.ts` | 8 Playwright UI tests covering user flows |
| `k6/stress-test.js` | K6 load test covering full user workflow |

## Test Results

### Rust Unit Tests: 13/13 Passed
Tests cover JWT creation/verification, error response mapping, User to UserResponse conversion, and password hash exclusion from serialized output.

### Rust Integration Tests: 20/20 Passed
Tests cover register, login, wrong password, create post, timeline, like/unlike, follow/unfollow, user profile, input validation (empty content, long content, invalid email), duplicate registration, delete post, user posts, followers/following lists, self-follow prevention, /me endpoint, unauthorized access, timeline with followed user posts, and double like conflict.

### Playwright UI Tests: 8/8 Passed
Tests cover register flow, register-to-login link, login flow, login-to-register link, post creation, like/unlike interaction, page navigation (home/profile/logout), and unauthenticated redirect.

### K6 Stress Test: 4030/4030 Checks Passed
- 0% failure rate
- p95 response time: 269ms
- 403 iterations completed
- 20 max virtual users
- Ramp pattern: 10s to 10 VUs, 20s at 20 VUs, 10s ramp down

## Architecture Decisions

1. **SQLite over PostgreSQL**: Chosen for zero-configuration setup. Single file database, no external server required. Suitable for the scope of this application.

2. **UUIDs as TEXT primary keys**: Allows client-side ID generation and avoids autoincrement contention. Stored as TEXT since SQLite lacks a native UUID type.

3. **JWT in localStorage**: Simpler than cookie-based auth. Provides CSRF protection since the token must be explicitly included in requests. Trade-off is XSS vulnerability if the application has script injection.

4. **TanStack Router over React Router**: Provides type-safe routing with TypeScript and built-in `beforeLoad` hooks for auth guards.

5. **TanStack Query for data fetching**: Provides caching, optimistic updates, and automatic query invalidation. Reduces boilerplate compared to manual fetch + state management.

6. **No ORM (raw SQLx queries)**: Keeps the dependency tree small and provides full control over queries. Appropriate for the relatively simple query patterns in this application.

7. **Argon2 for password hashing**: Industry-standard algorithm resistant to GPU attacks. Preferred over bcrypt for new applications.

8. **Composite primary keys on likes and follows**: Prevents duplicates at the database level without needing a separate unique constraint or ID column.
