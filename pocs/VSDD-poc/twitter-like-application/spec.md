# Twitter-Like Application — Specification

## Overview
A full-stack Twitter-like application with a Rust backend (Actix-web + SQLite) and React 19 frontend (TanStack Router, Vite, Bun). Features: user registration with default admin/admin, posts, likes, image uploads, search, timeline, profiles, and hot topics.

## Architecture

### Backend: Rust (latest stable)
- **Framework**: Actix-web 4
- **Database**: SQLite via rusqlite (with `PRAGMA foreign_keys = ON`)
- **Async Runtime**: Tokio
- **Image Storage**: Local filesystem (`uploads/` directory), UUID filename with original extension preserved (e.g., `uuid.jpg`)
- **Auth**: Session-based. Cookie name: `session_id`. Cookie attributes: HttpOnly, SameSite=Lax, Path=/. Session token is a random UUID stored in an in-memory HashMap. Sessions are lost on server restart.
- **JSON**: serde + serde_json. All endpoints accept/return `application/json` except post creation which uses `multipart/form-data`.
- **Password Hashing**: bcrypt with cost factor 12
- **CORS**: Backend sets `Access-Control-Allow-Origin` for `http://localhost:5173`, `Access-Control-Allow-Credentials: true`, and allows GET/POST/PUT/DELETE methods with Content-Type header.
- **Timestamps**: Unix epoch in seconds (i64).

### Frontend: React 19
- **Bundler**: Vite (dev mode with proxy to backend on port 8080)
- **Package Manager**: Bun
- **Routing**: TanStack Router
- **HTTP Client**: fetch API
- **Styling**: CSS

### Scripts
- `run.sh` — builds backend (cargo build --release), starts backend on port 8080 in background, installs frontend deps (bun install), starts frontend dev server on port 5173 in background. Writes PIDs to `.pids` file.
- `stop.sh` — reads `.pids` file and kills both processes.
- `test-all.sh` — runs `cargo test` for backend and `bun test` for frontend.

## Database Schema

```sql
PRAGMA foreign_keys = ON;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL,
    bio TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL
);

CREATE TABLE posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    author_id INTEGER NOT NULL REFERENCES users(id),
    content TEXT NOT NULL,
    image_url TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL
);

CREATE TABLE likes (
    user_id INTEGER NOT NULL REFERENCES users(id),
    post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
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

Default seed on startup: user `admin` with password `admin`, display_name `Admin`, bio empty string. Created only if no user with username `admin` exists.

## API Endpoints

All endpoints return JSON. Error responses: `{ "error": "message" }`.
User object shape: `{ id, username, display_name, bio, created_at }` (never includes password_hash).
Post object shape: `{ id, author_id, author_username, author_display_name, content, image_url, like_count, liked_by_me, created_at }`.

### Auth
- `POST /api/auth/register` — Body: `{ username, password, display_name }`. Returns 201 + user object + sets session cookie. Errors: 400 validation, 409 duplicate username.
- `POST /api/auth/login` — Body: `{ username, password }`. Returns 200 + user object + sets session cookie. Errors: 401 invalid credentials.
- `POST /api/auth/logout` — Clears session cookie. Returns 200. Works even if not authenticated.
- `GET /api/auth/me` — Returns 200 + user object. Errors: 401 if not authenticated.

### Posts
- `POST /api/posts` — Multipart form: `content` field (1-280 chars, required) + optional `image` file (JPEG/PNG only, max 5MB). Returns 201 + post object. Errors: 401 unauth, 400 validation (empty content, content too long, invalid image type, image too large).
- `GET /api/posts/:id` — Returns 200 + post object. `liked_by_me` is false if not authenticated. Errors: 404 not found.
- `DELETE /api/posts/:id` — Deletes post and associated likes (CASCADE). If post has an image, deletes the file from uploads/. Returns 204. Errors: 401 unauth, 403 not the author, 404 not found.
- `GET /api/posts` — Recent posts, sorted by created_at DESC. Query: `?page=1&limit=20`. Defaults: page=1, limit=20. Limit range: 1-100. Returns 200 + `{ posts: [...], total: N }`.

### Likes
- `POST /api/posts/:id/like` — Like a post. Idempotent. Returns 200 + `{ like_count }`. Errors: 401 unauth, 404 post not found.
- `DELETE /api/posts/:id/like` — Unlike a post. Idempotent. Returns 200 + `{ like_count }`. Errors: 401 unauth, 404 post not found.

### Follows
- `POST /api/users/:id/follow` — Follow a user. Idempotent. Returns 200. Errors: 401 unauth, 404 user not found, 400 self-follow.
- `DELETE /api/users/:id/follow` — Unfollow a user. Idempotent. Returns 200. Errors: 401 unauth, 404 user not found.
- `GET /api/users/:id/followers` — Returns 200 + `{ users: [...] }`. No pagination (return all).
- `GET /api/users/:id/following` — Returns 200 + `{ users: [...] }`. No pagination (return all).

### Timeline
- `GET /api/timeline` — Posts from followed users AND own posts, sorted by created_at DESC. Query: `?page=1&limit=20`. Defaults: page=1, limit=20. Limit range: 1-100. Ties broken by post id DESC. Returns 200 + `{ posts: [...], total: N }`. Returns empty list if no follows and no own posts. Errors: 401 unauth.

### Profile
- `GET /api/users/:id` — Public. Returns 200 + `{ id, username, display_name, bio, created_at, post_count, follower_count, following_count }`. Errors: 404 not found.
- `PUT /api/users/:id` — Body: `{ display_name, bio }`. Both fields required in request. display_name: 1-50 chars. bio: 0-200 chars. Returns 200 + updated user object. Errors: 401 unauth, 403 not own profile, 400 validation.
- `GET /api/users/:id/posts` — Posts by user, sorted by created_at DESC. Query: `?page=1&limit=20`. Returns 200 + `{ posts: [...], total: N }`. Errors: 404 user not found.

### Search
- `GET /api/search?q=term&type=posts` — Search posts by content using SQL LIKE with wildcards escaped (%, _, ' are escaped). Query: `?q=term&type=posts&page=1&limit=20`. Returns 200 + `{ posts: [...], total: N }`. Errors: 400 if q is empty or type is invalid.
- `GET /api/search?q=term&type=users` — Search users by username OR display_name using SQL LIKE with wildcards escaped. Returns 200 + `{ users: [...], total: N }`.

### Hot Topics
- `GET /api/hot` — Public. Returns top 10 posts with the most likes where post was created within the last 24 hours from the current server timestamp. Only posts with at least 1 like are included. Ties broken by created_at DESC then id DESC. Returns 200 + `{ posts: [...] }`. Returns empty list if no qualifying posts.

### Static Files
- `GET /uploads/:filename` — Serves uploaded images with correct Content-Type based on file extension.

## Edge Case Catalog

### Registration
- Empty username, password, or display_name -> 400
- Username > 30 chars -> 400
- Username with special chars (spaces, @, unicode) -> 400
- Duplicate username with different casing -> 409
- Password > 128 chars -> 400
- display_name > 50 chars -> 400

### Posts
- Empty content -> 400
- Content at exactly 280 chars -> 201 success
- Content at 281 chars -> 400
- Post with image > 5MB -> 400
- Post with non-image file (e.g., .txt) -> 400
- Post with valid JPEG -> 201 success
- Post with valid PNG -> 201 success
- Delete post that doesn't exist -> 404
- Delete another user's post -> 403
- Get post that doesn't exist -> 404
- Delete post with image -> image file deleted from disk

### Likes
- Like a post twice -> 200, count stays same
- Unlike a post not liked -> 200, count stays same
- Like non-existent post -> 404
- Like while not authenticated -> 401

### Follows
- Self-follow -> 400
- Follow twice -> 200, no duplicate
- Unfollow when not following -> 200
- Follow non-existent user -> 404

### Timeline
- No follows, no own posts -> empty list
- Follows but no posts from anyone -> empty list
- Pagination page beyond available -> empty list

### Search
- Empty search query -> 400
- Search with no results -> empty list
- Search with SQL wildcard chars (%, _) -> properly escaped, treated as literals
- Invalid type parameter -> 400

### Hot Topics
- No posts in last 24 hours -> empty list
- All posts have zero likes -> empty list
- Fewer than 10 qualifying posts -> return only those

## Non-Functional Requirements
- Backend serves on port 8080
- Frontend dev server on port 5173 with Vite proxy to backend
- SQLite database file: `data/twitter.db`
- Uploaded images served at `/uploads/filename`
- No user deletion feature (out of scope)
- No avatar upload feature (out of scope)
- Pagination is offset-based (simple, known limitation with concurrent writes)
- display_name allows any printable characters (no control characters or null bytes), 1-50 chars
- bio allows any printable characters, 0-200 chars

## Verification Architecture

### Property-Based Testing (proptest)
1. Any valid username (matching `[a-zA-Z0-9_]{1,30}`) passes registration validation
2. Any invalid username is rejected
3. Post content length validation is exact at 280 char boundary
4. Like idempotency — liking N times results in exactly 1 like row
5. Follow idempotency — following N times results in exactly 1 follow row
6. Timeline posts are always sorted by (created_at DESC, id DESC)
7. Search results for posts always contain the search term in content
8. Search results for users always contain the search term in username OR display_name

### Purity Boundary Map
- **Pure Core**: Validation functions (username, password, content length, image type/size check, display_name, bio), search term escaping, pagination offset calculation
- **Effectful Shell**: Database operations, file I/O (image storage), timestamp generation, session management, bcrypt hashing, HTTP handlers

### Test Categories
- **Unit Tests (Rust)**: Validation functions, pagination logic, search term escaping
- **Integration Tests (Rust)**: Full API endpoint tests using actix-web test server with test database
- **Frontend Tests (Bun)**: Component rendering tests
