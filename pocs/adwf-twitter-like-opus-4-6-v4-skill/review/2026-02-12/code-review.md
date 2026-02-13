# Code Review - Twitter-like App

**Date**: 2026-02-12

## Backend (Rust / Axum)

### Code Quality and Organization

The backend follows a clean modular structure: `main.rs`, `db.rs`, `auth.rs`, `errors.rs`, `models/`, and `handlers/`. Each domain concern has its own module. Handler files are organized by resource (users, posts, likes, follows, feed). The code is concise with no unnecessary abstractions.

The `PostRow` and `FeedRow` structs in `handlers/posts.rs` and `handlers/feed.rs` are duplicated. Both have identical fields and an identical `into_response()` method. This should be a single shared struct to avoid duplication.

### Error Handling

`AppError` is well-designed with five variants (NotFound, BadRequest, Unauthorized, Conflict, Internal) and proper HTTP status code mapping. Conversions from `sqlx::Error`, `bcrypt::BcryptError`, and `jsonwebtoken::errors::Error` are implemented via `From` traits. All handler functions return `Result<..., AppError>`, giving consistent JSON error responses.

One concern: `sqlx::Error` is mapped generically to `AppError::Internal(err.to_string())`. This exposes internal database error messages to the client. In production, these messages should be sanitized.

### API Design

The REST API is well-structured and follows conventional patterns:
- Resource-based routing (`/api/users`, `/api/posts`, `/api/feed`)
- Proper HTTP methods (GET, POST, DELETE)
- Pagination via query parameters with sensible defaults (page=1, limit=20, max=100)
- JWT-based authentication via `Authorization: Bearer` header
- The `AuthUser` extractor pattern cleanly separates authentication from business logic

The route patterns use `{id}` syntax which is the Axum 0.8+ convention.

Missing: no `GET /api/users/me` endpoint is documented in the design doc, but it exists in the code. The `getUserPosts` frontend function sends `user_id` as a query parameter, but the backend `get_all_posts` handler does not filter by `user_id` - it returns all posts regardless. This is a bug.

### Frontend Component Structure

Components are well-organized:
- **Pages**: LoginPage, RegisterPage, HomePage, ProfilePage, PostDetailPage
- **Components**: Navbar, PostCard, PostComposer, UserCard, FollowersList, Feed
- **API layer**: Separate files for each resource (client, users, posts, follows, likes)
- **Auth**: Context + hook pattern with `AuthProvider` and `useAuth`

The custom `Router` in `App.tsx` uses state-based routing instead of a library like React Router. This is simple and works but does not support browser back/forward navigation or URL sharing.

### Performance Considerations

- Like counts are computed via correlated subqueries `(SELECT COUNT(*) FROM likes WHERE post_id = p.id)` on every post fetch. For large datasets, this will become slow. An indexed counter column or materialized view would perform better.
- The `Feed` component uses TanStack Query's `useInfiniteQuery` for pagination, which is appropriate.
- `staleTime` is set to 30 seconds, which is reasonable for a social feed.
- SQLite connection pool is limited to 5 connections. This is adequate for SQLite but may need tuning under load.
- bcrypt cost factor is 12, which is secure but adds latency to register/login operations.

### Naming Conventions

Naming is consistent throughout:
- Rust: snake_case for functions and variables, PascalCase for types
- TypeScript: camelCase for functions and variables, PascalCase for components and types
- API endpoints use plural nouns with kebab-case where needed

### Bugs and Issues Found

1. **getUserPosts does not filter by user**: The frontend `getUserPosts(userId, page)` calls `/api/posts?user_id=${userId}&page=${page}`, but the backend `get_all_posts` handler ignores the `user_id` query parameter entirely. The ProfilePage posts tab will show all posts from all users, not just the profile user's posts.

2. **liked_by_me is never set by the backend**: The `Post` type on the frontend has `liked_by_me?: boolean`, and `PostCard` and `PostDetailPage` use it to toggle the like button state. However, the backend `PostResponse` has no `liked_by_me` field and never returns this value. The like toggle button will always show the "not liked" state.

3. **PaginatedResponse type is defined but unused**: The frontend defines `PaginatedResponse<T>` with `data`, `page`, `per_page`, `total` fields, but the backend returns plain arrays, not this envelope format. The type is dead code.

4. **FollowInfo type is defined but unused**: Similarly, `FollowInfo` with `followers_count`, `following_count`, `is_following` is defined but never used anywhere.

5. **LikeCount type mismatch**: The frontend `LikeCount` type expects `{ count, liked_by_me }`, but the backend `LikeCountResponse` returns `{ post_id, count }` with no `liked_by_me` field.

6. **PostRow / FeedRow duplication**: Identical struct definitions in `handlers/posts.rs` and `handlers/feed.rs`.

7. **delete_post should return 403 Forbidden**: When a user tries to delete another user's post, the handler returns `Unauthorized` (401). The correct status code is `Forbidden` (403) since the user is authenticated but lacks permission.
