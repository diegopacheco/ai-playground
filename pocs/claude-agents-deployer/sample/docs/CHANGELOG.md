# Changelog

## v1.0.0 - Blog Platform

Full-stack blog platform with a Rust backend, React frontend, and PostgreSQL database.

### Backend (Rust)

- Axum 0.8 web framework with Tokio async runtime
- RESTful API with full CRUD operations for posts, comments, and users
- SQLx 0.8 with PostgreSQL for data persistence
- UUID v4 primary keys generated server-side
- camelCase JSON serialization via `serde(rename_all = "camelCase")` on all response structs (Post, Comment, User)
- CORS support via tower-http allowing any origin, method, and header
- Tracing with env-filter for structured logging
- Auto-creates tables on startup with `CREATE TABLE IF NOT EXISTS`
- Server binds to `0.0.0.0:8080`
- API endpoints:
  - `POST /api/users` - create user
  - `GET /api/users` - list all users
  - `GET /api/users/{id}` - get user by UUID
  - `POST /api/posts` - create post
  - `GET /api/posts` - list all posts (ordered by created_at DESC)
  - `GET /api/posts/{id}` - get post by UUID
  - `PUT /api/posts/{id}` - update post (partial update supported)
  - `DELETE /api/posts/{id}` - delete post
  - `POST /api/posts/{post_id}/comments` - create comment on post
  - `GET /api/posts/{post_id}/comments` - list comments for post
  - `DELETE /api/comments/{id}` - delete comment

### Frontend (React)

- React 19 with TypeScript
- TanStack Router 1.92 for client-side routing with type-safe params
- TanStack Query 5.62 for server state management and data fetching
- Tailwind CSS 4 for utility-first styling
- Vite 6 with React plugin for build tooling
- All IDs typed as `string` to match UUID responses from backend
- Custom `useApi` hook for centralized API calls
- Pages:
  - Home (`/`) - lists all posts
  - Post Detail (`/posts/$postId`) - single post view with comments
  - Create Post (`/posts/create`) - form to create new post
  - Edit Post (`/posts/$postId/edit`) - form to update existing post
  - Profile (`/profile`) - user list page
- Navigation bar with links to Home, New Post, and Profile

### Database (PostgreSQL 17)

- Three tables: `users`, `posts`, `comments`
- `users` table: id (UUID PK), name, email (UNIQUE), created_at
- `posts` table: id (UUID PK), title, content, author, created_at, updated_at
- `comments` table: id (UUID PK), content, author, post_id (FK to posts), created_at
- Foreign key constraint on comments.post_id with `ON DELETE CASCADE`
- Indexes:
  - `idx_posts_created_at` on posts(created_at DESC)
  - `idx_comments_post_id` on comments(post_id)
  - `idx_comments_created_at` on comments(created_at DESC)
  - `idx_users_email` on users(email)
- Podman container scripts:
  - `start-db.sh` - starts PostgreSQL 17 container with readiness loop
  - `stop-db.sh` - stops and removes the container
  - `create-schema.sh` - runs schema.sql against the database
  - `run-sql-client.sh` - opens psql client session

### Testing

- 11 Rust unit tests covering model serialization and deserialization:
  - `test_create_post_deserialize` - validates CreatePost JSON parsing
  - `test_update_post_partial` - validates partial UpdatePost with optional fields
  - `test_update_post_all_fields` - validates full UpdatePost parsing
  - `test_post_serialize_camel_case` - confirms camelCase output for Post
  - `test_post_roundtrip` - serialize then deserialize Post equality
  - `test_create_user_deserialize` - validates CreateUser JSON parsing
  - `test_user_serialize_camel_case` - confirms camelCase output for User
  - `test_user_roundtrip` - serialize then deserialize User equality
  - `test_create_comment_deserialize` - validates CreateComment JSON parsing
  - `test_comment_serialize_camel_case` - confirms camelCase output for Comment
  - `test_comment_roundtrip` - serialize then deserialize Comment equality
- 15 integration tests using axum test utilities (tower::ServiceExt oneshot):
  - `test_create_user` - POST /api/users returns 201
  - `test_list_users` - GET /api/users returns array
  - `test_get_user_by_id` - GET /api/users/{id} returns correct user
  - `test_get_user_not_found` - GET /api/users/{nil} returns 404
  - `test_create_post` - POST /api/posts returns 201
  - `test_list_posts` - GET /api/posts returns array
  - `test_get_post_by_id` - GET /api/posts/{id} returns correct post
  - `test_update_post` - PUT /api/posts/{id} with partial body returns updated post
  - `test_delete_post` - DELETE /api/posts/{id} returns 200, subsequent GET returns 404
  - `test_delete_post_not_found` - DELETE /api/posts/{nil} returns 404
  - `test_create_comment` - POST /api/posts/{id}/comments returns 201
  - `test_list_comments` - GET /api/posts/{id}/comments returns array of 2
  - `test_delete_comment` - DELETE /api/comments/{id} returns 200
  - `test_create_comment_on_nonexistent_post` - returns 404 with error
  - `test_cascade_delete_post_removes_comments` - deleting post empties its comments
- 7 Playwright E2E tests:
  - `home page loads` - verifies Blog Platform text
  - `nav bar has links` - checks Home, New Post, Profile links
  - `navigate to create post page` - clicks New Post, checks URL
  - `create post form has fields` - verifies title input exists
  - `navigate to profile page` - clicks Profile, checks URL
  - `home page shows no posts message when empty` - checks empty state
  - `navigate back to home from create post` - clicks Home, checks URL
- K6 performance tests:
  - `smoke-test.js` - 2 VUs for 30s, p95 < 500ms, error rate < 1%
  - `load-test.js` - ramp to 50 VUs over 2 minutes, p95 < 500ms, error rate < 1%
  - `stress-test.js` - ramp to 150 VUs over 3 minutes, p95 < 1000ms, error rate < 5%

### Review Fixes Applied

- Fixed frontend-backend type mismatch: changed all frontend ID types from `number` to `string` to match UUID responses
- Added `#[serde(rename_all = "camelCase")]` to all Rust response structs (Post, Comment, User)
- Removed non-existent fields (excerpt, bio, avatarUrl) from frontend TypeScript types
- Replaced broken `/api/users/me` endpoint with `/api/users` list endpoint on Profile page
- Synced `db/schema.sql` to match the actual runtime schema used in `main.rs`
