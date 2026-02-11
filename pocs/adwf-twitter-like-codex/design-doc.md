# Twitter-like Web Clone Design

## Architecture overview
- Frontend: React single-page app served as static files.
- Backend: Rust with Axum providing JSON REST APIs.
- Database: SQLite with SQL migrations for users, posts, likes, and follows.
- Integration: Frontend calls backend APIs over HTTP; backend persists to SQLite.
- Local runtime ports: backend `3001`, frontend `4173`.

## Backend API endpoints and responsibilities
- `GET /health`: service health.
- `POST /api/users`: create user.
- `POST /api/follows`: create follow relationship.
- `POST /api/posts`: create post.
- `GET /api/posts`: list recent posts with author and like count.
- `POST /api/posts/:id/likes`: like a post.
- `GET /api/timeline/:user_id`: list timeline posts from followed users and self.

## Frontend components and interactions
- `App`: page shell and state owner.
- `CreateUserForm`: create a local active user.
- `CreatePostForm`: create posts as active user.
- `Timeline`: render posts feed sorted by recency.
- `PostCard`: render post text, author, created date, likes, and like action.

## Database schema design
- `users(id, username, created_at)`
- `posts(id, user_id, content, created_at)`
- `likes(user_id, post_id, created_at)` with unique `(user_id, post_id)`
- `follows(follower_id, followee_id, created_at)` with unique `(follower_id, followee_id)`

## Integration points between frontend, backend, database
- Frontend calls backend with `fetch` against `/api/*`.
- Backend validates input and executes SQL using `sqlx`.
- Migrations run before backend startup to ensure schema availability.
- Verification scripts: `tests/integration.sh`, `tests/ui.sh`, `tests/stress.sh`.
