# Design Doc - Twitter-like App

## Architecture Overview

A Twitter-like social media platform with posts (tweets), likes, follows, and a user feed. The system uses a Rust backend with Axum, a React 19 frontend with TypeScript, and SQLite for persistence.

### Stack
- **Backend**: Rust 1.93+, Axum 0.8, SQLx 0.8, Tokio, Serde, SQLite, bcrypt 0.17, jsonwebtoken 9, chrono
- **Frontend**: React 19, TypeScript, Vite, Bun, TanStack Query, Tailwind CSS
- **Database**: SQLite (file-based, no container needed)

### Architecture Diagram
```
[React Frontend :5173] --> [Rust Backend :8080] --> [SQLite DB]
```

## Backend API Endpoints

### Users
- `POST /api/users` - Register a new user (username, email, password). Returns a LoginResponse with JWT token and user object.
- `POST /api/users/login` - Login and get token
- `GET /api/users/me` - Get current authenticated user profile
- `GET /api/users/{id}` - Get user profile by ID
- `GET /api/users/{id}/followers` - List user followers
- `GET /api/users/{id}/following` - List who user follows

### Posts (Tweets)
- `POST /api/posts` - Create a new post (content, max 280 chars). Requires auth.
- `GET /api/posts/{id}` - Get a single post
- `DELETE /api/posts/{id}` - Delete own post. Requires auth.
- `GET /api/posts` - Get all posts (paginated via `page` and `limit` query params, newest first)

### Feed
- `GET /api/feed` - Get personalized feed (posts from followed users and own posts, paginated). Requires auth.

### Likes
- `POST /api/posts/{id}/like` - Like a post. Requires auth.
- `DELETE /api/posts/{id}/like` - Unlike a post. Requires auth.
- `GET /api/posts/{id}/likes` - Get like count

### Follows
- `POST /api/users/{id}/follow` - Follow a user. Requires auth.
- `DELETE /api/users/{id}/follow` - Unfollow a user. Requires auth.

## Frontend Components

### Pages
- **LoginPage** - Login form with username/password
- **RegisterPage** - Registration form. On success, auto-logs in and redirects to home.
- **HomePage** - Main feed with post composer at top
- **ProfilePage** - User profile with their posts, follower/following tabs with counts
- **PostDetailPage** - Single post view with likes

### Components
- **PostCard** - Displays a single post with author avatar, username, content, relative timestamp, like button, like count
- **PostComposer** - Text area with character counter (280 max) and submit button
- **UserCard** - User avatar, username, follow/unfollow button (hidden for own profile)
- **Feed** - Scrollable list of PostCards with infinite scroll via TanStack Query `useInfiniteQuery` and "Load More" button
- **Navbar** - Top nav with app name "Chirp", home link, profile link, username display, logout button
- **FollowersList** - List of UserCards for followers/following

### State Management
- TanStack Query for server state (posts, users, feed)
- React Context for auth state (token, current user)
- Custom state-based router in App.tsx (no URL routing library)

## Database Schema

### users
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| username | TEXT | NOT NULL UNIQUE |
| email | TEXT | NOT NULL UNIQUE |
| password_hash | TEXT | NOT NULL |
| created_at | TEXT | NOT NULL DEFAULT (datetime('now')) |

### posts
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| user_id | INTEGER | NOT NULL REFERENCES users(id) |
| content | TEXT | NOT NULL CHECK(length(content) <= 280) |
| created_at | TEXT | NOT NULL DEFAULT (datetime('now')) |

### likes
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| user_id | INTEGER | NOT NULL REFERENCES users(id) |
| post_id | INTEGER | NOT NULL REFERENCES posts(id) |
| created_at | TEXT | NOT NULL DEFAULT (datetime('now')) |
| | | UNIQUE(user_id, post_id) |

### follows
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| follower_id | INTEGER | NOT NULL REFERENCES users(id) |
| following_id | INTEGER | NOT NULL REFERENCES users(id) |
| created_at | TEXT | NOT NULL DEFAULT (datetime('now')) |
| | | UNIQUE(follower_id, following_id) |
| | | CHECK(follower_id != following_id) |

## Integration Points

- Frontend authenticates via `POST /api/users/login` (or `POST /api/users` for registration) which returns a JWT token and user object
- All authenticated endpoints require `Authorization: Bearer <token>` header
- Frontend uses TanStack Query to cache and sync data with backend
- Backend uses SQLx with runtime queries (not compile-time macros) to avoid needing DATABASE_URL at compile time
- CORS configured on backend to allow frontend origin (http://localhost:5173) with permissive methods and headers
- Backend serves on port 8080, frontend dev server on port 5173
- Token and user data persisted in localStorage for session persistence across page reloads
- JWT tokens expire after 24 hours
