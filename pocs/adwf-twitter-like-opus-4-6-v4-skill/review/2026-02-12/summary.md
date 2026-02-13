# Changes Summary - Twitter-like App

**Date**: 2026-02-12

## What Was Built

A full-stack Twitter-like social media application named "Chirp" with the following capabilities:
- User registration and login with JWT authentication
- Post creation (280 character limit) and deletion
- Like and unlike posts
- Follow and unfollow users
- Personalized feed showing posts from followed users and own posts
- User profile pages with posts, followers, and following tabs
- Infinite scroll pagination

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backend framework | Axum 0.8 with Tokio | Async, type-safe Rust web framework |
| Database | SQLite via SQLx (runtime queries) | No external database server needed; avoids compile-time DATABASE_URL requirement |
| Authentication | JWT with HS256, 24h expiry | Stateless auth, no session storage needed |
| Password hashing | bcrypt cost 12 | Industry standard with reasonable compute cost |
| Frontend framework | React 19 with TypeScript | Modern component-based UI with type safety |
| State management | TanStack Query + React Context | Server state caching via TanStack Query; auth state via Context |
| Styling | Tailwind CSS | Utility-first, no custom CSS files needed |
| Build tooling | Vite + Bun | Fast dev server and package management |
| Routing | Custom state-based router | No library dependency; simple but no URL support |
| Migration | Inline CREATE TABLE IF NOT EXISTS | No migration tool needed; tables created at startup |

## Test Coverage

| Category | Count | Location |
|----------|-------|----------|
| Backend unit tests | 23 | Inline `#[cfg(test)]` modules in `auth.rs`, `errors.rs`, `models/user.rs`, `models/post.rs`, `models/like.rs` |
| Frontend unit tests | 19 | `frontend/src/components/__tests__/` and `frontend/src/pages/__tests__/` |
| Integration tests | 13 | `integration-tests/run-integration-tests.sh` (curl-based) |
| E2E tests | 3 | `e2e-tests/tests/app.spec.ts` (Playwright) |
| K6 stress tests | 1 script | `stress-tests/load-test.js` (ramp to 10 VUs, 35s duration) |

Backend unit tests cover: token creation/validation, error code mapping, model serialization/deserialization, password hash exclusion.

Frontend unit tests cover: PostComposer, PostCard, Navbar, UserCard, LoginPage, RegisterPage rendering and interactions.

Integration tests cover: register, login, create post, get posts, get single post, like, get like count, register second user, follow, get feed, unlike, unfollow, delete post.

E2E tests cover: login page load, navigate to register, register and login flow.

K6 stress test: ramps to 10 virtual users, each registering, logging in, creating a post, fetching posts, and fetching feed. Thresholds: p95 response time under 2000ms, error rate under 10%.

## Key Files

### Backend
- `backend/src/main.rs` - Application entry point, route definitions, CORS setup
- `backend/src/db.rs` - Database pool creation and table migrations
- `backend/src/auth.rs` - JWT token creation, validation, AuthUser extractor
- `backend/src/errors.rs` - AppError enum and HTTP response mapping
- `backend/src/models/user.rs` - User, UserResponse, CreateUserRequest, LoginRequest, LoginResponse
- `backend/src/models/post.rs` - Post, PostResponse, CreatePostRequest, PaginationParams
- `backend/src/models/like.rs` - Like, LikeCountResponse
- `backend/src/models/follow.rs` - Follow
- `backend/src/handlers/users.rs` - Register, login, get user, get followers/following, get me
- `backend/src/handlers/posts.rs` - Create, get, delete, list posts
- `backend/src/handlers/likes.rs` - Like, unlike, get like count
- `backend/src/handlers/follows.rs` - Follow, unfollow
- `backend/src/handlers/feed.rs` - Personalized feed

### Frontend
- `frontend/src/App.tsx` - Root component, QueryClient, Router
- `frontend/src/main.tsx` - React DOM entry point
- `frontend/src/types/index.ts` - TypeScript type definitions
- `frontend/src/context/AuthContext.tsx` - Auth state provider
- `frontend/src/hooks/useAuth.ts` - Auth context hook
- `frontend/src/api/client.ts` - Base API client with auth header injection
- `frontend/src/api/users.ts` - User API functions
- `frontend/src/api/posts.ts` - Post API functions
- `frontend/src/api/follows.ts` - Follow API functions
- `frontend/src/api/likes.ts` - Like API functions
- `frontend/src/components/Navbar.tsx` - Navigation bar
- `frontend/src/components/PostCard.tsx` - Post display card
- `frontend/src/components/PostComposer.tsx` - New post form
- `frontend/src/components/UserCard.tsx` - User display card
- `frontend/src/components/FollowersList.tsx` - Followers/following list
- `frontend/src/components/Feed.tsx` - Infinite scroll feed
- `frontend/src/pages/LoginPage.tsx` - Login page
- `frontend/src/pages/RegisterPage.tsx` - Registration page
- `frontend/src/pages/HomePage.tsx` - Main feed page
- `frontend/src/pages/ProfilePage.tsx` - User profile page
- `frontend/src/pages/PostDetailPage.tsx` - Single post detail page

### Tests and Infrastructure
- `integration-tests/run-integration-tests.sh` - Bash integration test suite
- `e2e-tests/tests/app.spec.ts` - Playwright E2E tests
- `e2e-tests/playwright.config.ts` - Playwright configuration
- `stress-tests/load-test.js` - K6 load test script
- `run-e2e-tests.sh` - E2E test runner script

## Known Issues

1. **getUserPosts is broken**: The frontend calls `/api/posts?user_id=${userId}` but the backend ignores the `user_id` parameter. Profile page posts tab shows all posts instead of only the user's posts.
2. **liked_by_me never populated**: The backend does not return `liked_by_me` in post responses, so the frontend like toggle always shows the "not liked" state.
3. **Hardcoded JWT secret**: The secret `"twitter-like-super-secret-key-2024"` is hardcoded in source code.
4. **No rate limiting**: No protection against brute force or spam.
5. **Unused frontend types**: `PaginatedResponse`, `FollowInfo`, and `LikeCount.liked_by_me` are defined but never match backend responses.
6. **PostRow/FeedRow duplication**: Identical structs in two handler files.
7. **No URL-based routing**: Browser back/forward buttons and URL sharing do not work.
8. **Internal errors leak details**: SQLx error messages are forwarded to clients.
9. **delete_post returns 401 instead of 403**: When a user attempts to delete another user's post.
