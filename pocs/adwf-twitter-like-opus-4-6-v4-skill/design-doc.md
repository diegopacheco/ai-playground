# Design Doc - Twitter-like App

## Architecture Overview

A Twitter-like social media platform with posts (tweets), likes, follows, and a user feed. The system uses a Rust backend with Axum, a React 19 frontend with TypeScript, and SQLite for persistence.

### Stack
- **Backend**: Rust 1.93+, Axum, SQLx, Tokio, Serde, SQLite
- **Frontend**: React 19, TypeScript, Vite, Bun, TanStack Query, Tailwind CSS
- **Database**: SQLite (file-based, no container needed)

### Architecture Diagram
```
[React Frontend :5173] --> [Rust Backend :8080] --> [SQLite DB]
```

## Backend API Endpoints

### Users
- `POST /api/users` - Register a new user (username, email, password)
- `POST /api/users/login` - Login and get token
- `GET /api/users/:id` - Get user profile
- `GET /api/users/:id/followers` - List user followers
- `GET /api/users/:id/following` - List who user follows

### Posts (Tweets)
- `POST /api/posts` - Create a new post (content, max 280 chars)
- `GET /api/posts/:id` - Get a single post
- `DELETE /api/posts/:id` - Delete own post
- `GET /api/posts` - Get all posts (paginated, newest first)

### Feed
- `GET /api/feed` - Get personalized feed (posts from followed users, paginated)

### Likes
- `POST /api/posts/:id/like` - Like a post
- `DELETE /api/posts/:id/like` - Unlike a post
- `GET /api/posts/:id/likes` - Get like count

### Follows
- `POST /api/users/:id/follow` - Follow a user
- `DELETE /api/users/:id/follow` - Unfollow a user

## Frontend Components

### Pages
- **LoginPage** - Login form with username/password
- **RegisterPage** - Registration form
- **HomePage** - Main feed with post composer at top
- **ProfilePage** - User profile with their posts, follower/following counts
- **PostDetailPage** - Single post view with likes

### Components
- **PostCard** - Displays a single post with author, content, timestamp, like button, like count
- **PostComposer** - Text area with character counter (280 max) and submit button
- **UserCard** - User avatar, username, follow/unfollow button
- **Feed** - Scrollable list of PostCards with infinite scroll (TanStack Query)
- **Navbar** - Top nav with logo, home link, profile link, logout
- **FollowersList** - List of UserCards for followers/following

### State Management
- TanStack Query for server state (posts, users, feed)
- React Context for auth state (token, current user)

## Database Schema

### users
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| username | TEXT | NOT NULL UNIQUE |
| email | TEXT | NOT NULL UNIQUE |
| password_hash | TEXT | NOT NULL |
| created_at | TEXT | NOT NULL DEFAULT CURRENT_TIMESTAMP |

### posts
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| user_id | INTEGER | NOT NULL REFERENCES users(id) |
| content | TEXT | NOT NULL CHECK(length(content) <= 280) |
| created_at | TEXT | NOT NULL DEFAULT CURRENT_TIMESTAMP |

### likes
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| user_id | INTEGER | NOT NULL REFERENCES users(id) |
| post_id | INTEGER | NOT NULL REFERENCES posts(id) |
| created_at | TEXT | NOT NULL DEFAULT CURRENT_TIMESTAMP |
| | | UNIQUE(user_id, post_id) |

### follows
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| follower_id | INTEGER | NOT NULL REFERENCES users(id) |
| following_id | INTEGER | NOT NULL REFERENCES users(id) |
| created_at | TEXT | NOT NULL DEFAULT CURRENT_TIMESTAMP |
| | | UNIQUE(follower_id, following_id) |
| | | CHECK(follower_id != following_id) |

## Integration Points

- Frontend authenticates via `POST /api/users/login` which returns a JWT token
- All authenticated endpoints require `Authorization: Bearer <token>` header
- Frontend uses TanStack Query to cache and sync data with backend
- Backend uses SQLx with compile-time checked queries against SQLite
- CORS configured on backend to allow frontend origin (localhost:5173)
- Backend serves on port 8080, frontend dev server on port 5173
