# Twitter-Like Application — Specification

## Overview
A full-stack Twitter-like application with a Rust backend (Actix-web + SQLite) and React 19 frontend (TanStack Router, Vite, Bun). Features: user registration, authentication, posts, likes, image uploads, search, timeline, profiles, and hot topics.

## Architecture

### Backend: Rust 1.93+
- **Framework**: Actix-web 4
- **Database**: SQLite via rusqlite
- **Async Runtime**: Tokio
- **Image Storage**: Local filesystem (uploads/ directory)
- **Auth**: Session-based with cookie tokens
- **JSON**: serde + serde_json

### Frontend: React 19
- **Bundler**: Vite
- **Package Manager**: Bun
- **Routing**: TanStack Router
- **HTTP Client**: fetch API (no axios)
- **Styling**: CSS (no framework)

### Scripts
- `run.sh` — builds and starts both backend and frontend
- `stop.sh` — stops both backend and frontend
- `test-all.sh` — runs all backend and frontend tests

## Database Schema

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL,
    bio TEXT DEFAULT '',
    avatar_url TEXT DEFAULT '',
    created_at INTEGER NOT NULL
);

CREATE TABLE posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    author_id INTEGER NOT NULL REFERENCES users(id),
    content TEXT NOT NULL,
    image_url TEXT DEFAULT '',
    created_at INTEGER NOT NULL
);

CREATE TABLE likes (
    user_id INTEGER NOT NULL REFERENCES users(id),
    post_id INTEGER NOT NULL REFERENCES posts(id),
    created_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, post_id)
);

CREATE TABLE follows (
    follower_id INTEGER NOT NULL REFERENCES users(id),
    followee_id INTEGER NOT NULL REFERENCES users(id),
    created_at INTEGER NOT NULL,
    PRIMARY KEY (follower_id, followee_id),
    CHECK (follower_id != followee_id)
);
```

Default seed: user `admin` with password `admin` is created on first startup.

## API Endpoints

### Auth
- `POST /api/auth/register` — Register new user. Body: `{ username, password, display_name }`. Username: 1-30 chars, ASCII alphanumeric + underscore. Password: 1-128 chars. Returns user object + sets session cookie.
- `POST /api/auth/login` — Login. Body: `{ username, password }`. Returns user object + sets session cookie.
- `POST /api/auth/logout` — Logout. Clears session cookie.
- `GET /api/auth/me` — Get current authenticated user.

### Posts
- `POST /api/posts` — Create post. Multipart form: `content` (1-280 chars) + optional `image` (JPEG/PNG, max 5MB). Returns created post.
- `GET /api/posts/:id` — Get single post with author info and like count.
- `DELETE /api/posts/:id` — Delete own post. Only author can delete.
- `GET /api/posts` — List recent posts (paginated). Query: `?page=1&limit=20`.

### Likes
- `POST /api/posts/:id/like` — Like a post. Idempotent (liking twice is not an error, just no-op).
- `DELETE /api/posts/:id/like` — Unlike a post. Idempotent.
- `GET /api/posts/:id/likes` — Get like count and whether current user liked it.

### Follows
- `POST /api/users/:id/follow` — Follow a user. Cannot self-follow. Idempotent.
- `DELETE /api/users/:id/follow` — Unfollow a user. Idempotent.
- `GET /api/users/:id/followers` — Get followers list.
- `GET /api/users/:id/following` — Get following list.

### Timeline
- `GET /api/timeline` — Get posts from followed users, sorted by most recent. Query: `?page=1&limit=20`. Includes the authenticated user's own posts.

### Profile
- `GET /api/users/:id` — Get user profile (display_name, bio, avatar_url, post count, follower count, following count).
- `PUT /api/users/:id` — Update own profile. Body: `{ display_name, bio }`. Only the user themselves can update.
- `GET /api/users/:id/posts` — Get posts by a specific user (paginated).

### Search
- `GET /api/search?q=term&type=posts` — Search posts by content (LIKE match). Query: `?q=term&type=posts&page=1&limit=20`.
- `GET /api/search?q=term&type=users` — Search users by username or display_name.

### Hot Topics
- `GET /api/hot` — Returns top 10 posts with most likes in the last 24 hours.

## Behavioral Contracts

### Registration
- **Preconditions**: username non-empty, 1-30 chars, ASCII `[a-zA-Z0-9_]` only, unique (case-insensitive). Password non-empty, 1-128 chars. display_name non-empty, 1-50 chars.
- **Postconditions**: User stored in DB. Password stored as bcrypt hash. Session cookie set. Returns user object (no password_hash).
- **Errors**: 400 for validation failures, 409 for duplicate username.

### Login
- **Preconditions**: username and password provided.
- **Postconditions**: Session cookie set if credentials valid.
- **Errors**: 401 for invalid credentials.

### Post Creation
- **Preconditions**: User authenticated. Content 1-280 chars. Image optional, must be JPEG/PNG, max 5MB.
- **Postconditions**: Post stored in DB. Image saved to `uploads/` with UUID filename. Returns post with generated id and timestamp.
- **Errors**: 401 if not authenticated. 400 for validation failures.

### Like/Unlike
- **Preconditions**: User authenticated. Post must exist.
- **Postconditions**: Like row inserted/deleted. Idempotent — no error if already liked/not liked.
- **Errors**: 401 if not authenticated. 404 if post not found.

### Follow/Unfollow
- **Preconditions**: User authenticated. Target user must exist. Cannot self-follow.
- **Postconditions**: Follow row inserted/deleted. Idempotent.
- **Errors**: 401 if not authenticated. 404 if target user not found. 400 if self-follow.

### Timeline
- **Preconditions**: User authenticated. Page >= 1, limit 1-100.
- **Postconditions**: Returns posts from followed users + own posts, sorted by created_at DESC, paginated.
- **Errors**: 401 if not authenticated.

### Search
- **Preconditions**: `q` parameter non-empty. `type` is "posts" or "users".
- **Postconditions**: Returns matching results via SQL LIKE `%term%`, paginated.
- **Errors**: 400 if q is empty or type is invalid.

### Hot Topics
- **Preconditions**: None (public endpoint).
- **Postconditions**: Returns top 10 posts by like count where post created_at is within last 24 hours. Includes author info and like count.

### Profile Update
- **Preconditions**: User authenticated. Can only update own profile.
- **Postconditions**: display_name and bio updated in DB.
- **Errors**: 401 if not authenticated. 403 if updating another user's profile.

## Edge Case Catalog

### Registration
- Empty username, password, or display_name
- Username > 30 chars
- Username with special chars (spaces, @, unicode)
- Duplicate username with different casing
- Password > 128 chars
- display_name > 50 chars

### Posts
- Empty content
- Content at exactly 280 chars
- Content at 281 chars
- Post with image > 5MB
- Post with non-image file
- Post with valid JPEG and PNG
- Delete post that doesn't exist
- Delete another user's post
- Get post that doesn't exist

### Likes
- Like a post twice (idempotent)
- Unlike a post not liked (idempotent)
- Like non-existent post
- Like while not authenticated

### Follows
- Self-follow
- Follow twice (idempotent)
- Unfollow when not following (idempotent)
- Follow non-existent user

### Timeline
- Timeline with no follows (only own posts)
- Timeline with follows who have no posts
- Pagination beyond available posts

### Search
- Empty search query
- Search with no results
- Search with special SQL chars (%, _, ')
- Invalid type parameter

### Hot Topics
- No posts in last 24 hours (empty result)
- Posts with zero likes excluded from results

## Non-Functional Requirements
- Backend serves on port 8080
- Frontend serves on port 5173 (Vite dev) or built static files served by backend
- SQLite database file: `data/twitter.db`
- Uploaded images served at `/uploads/filename`
- Password hashing: bcrypt
- Session tokens: random UUID stored in memory (HashMap)
- No external services required

## Verification Architecture

### Property-Based Testing (proptest)
1. Any valid username passes registration, any invalid username is rejected
2. Post content length validation is exact at 280 char boundary
3. Like idempotency — liking N times results in exactly 1 like
4. Follow idempotency — following N times results in exactly 1 follow
5. Timeline always returns posts sorted by created_at DESC
6. Search results always contain the search term

### Purity Boundary Map
- **Pure Core**: Validation functions (username, password, content, image type/size), search term sanitization, pagination math
- **Effectful Shell**: Database operations, file I/O (image storage), timestamp generation, session management, HTTP handlers

### Test Categories
- **Unit Tests**: Validation functions, pagination logic, password hashing
- **Integration Tests**: Full API endpoint tests using actix-web test utilities
- **Frontend Tests**: Component rendering, API integration mocks
