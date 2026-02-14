# Design Doc - Twitter-like Application

## Architecture Overview

A Twitter-like social media application with three tiers:
- **Frontend**: React 19 + TypeScript + Vite + Tailwind CSS + TanStack Query
- **Backend**: Rust (Axum) REST API with SQLx for database access
- **Database**: PostgreSQL 18 in a container

The frontend communicates with the backend via REST API. The backend connects to PostgreSQL for data persistence. Authentication uses JWT tokens with bcrypt password hashing.

## Backend API Endpoints

### Auth
- `POST /api/auth/register` - Register new user (username, email, password) -> returns JWT
- `POST /api/auth/login` - Login (email, password) -> returns JWT

### Users
- `GET /api/users/:id` - Get user profile
- `GET /api/users/:id/followers` - Get followers list
- `GET /api/users/:id/following` - Get following list
- `PUT /api/users/:id` - Update profile (display_name, bio) [auth required]

### Tweets
- `POST /api/tweets` - Create tweet (content, max 280 chars) [auth required]
- `GET /api/tweets/:id` - Get single tweet
- `DELETE /api/tweets/:id` - Delete tweet [auth required, owner only]
- `GET /api/tweets/feed` - Get home feed (tweets from followed users) [auth required]
- `GET /api/users/:id/tweets` - Get user's tweets

### Likes
- `POST /api/tweets/:id/like` - Like a tweet [auth required]
- `DELETE /api/tweets/:id/like` - Unlike a tweet [auth required]

### Follows
- `POST /api/users/:id/follow` - Follow a user [auth required]
- `DELETE /api/users/:id/follow` - Unfollow a user [auth required]

## Frontend Components

### Pages
- **LoginPage** - Email/password login form
- **RegisterPage** - Username/email/password registration form
- **HomePage** - Feed of tweets from followed users, compose tweet box
- **ProfilePage** - User profile with their tweets, follower/following counts, follow/unfollow button
- **TweetDetailPage** - Single tweet view

### Shared Components
- **TweetCard** - Displays tweet with author info, content, like count, like button, delete button (if owner)
- **UserCard** - Displays user info with follow/unfollow button
- **NavBar** - Top navigation with logo, home link, profile link, logout
- **ComposeBox** - Text area with character counter (280 max) and post button
- **ProtectedRoute** - Redirects to login if not authenticated

### State Management
- TanStack Query for server state (tweets, users, followers)
- React Context for auth state (JWT token, current user)

## Database Schema

### users
| Column | Type | Constraints |
|--------|------|-------------|
| id | SERIAL | PRIMARY KEY |
| username | VARCHAR(50) | UNIQUE, NOT NULL |
| email | VARCHAR(255) | UNIQUE, NOT NULL |
| password_hash | VARCHAR(255) | NOT NULL |
| display_name | VARCHAR(100) | NOT NULL |
| bio | TEXT | DEFAULT '' |
| created_at | TIMESTAMP | DEFAULT NOW() |

### tweets
| Column | Type | Constraints |
|--------|------|-------------|
| id | SERIAL | PRIMARY KEY |
| user_id | INTEGER | REFERENCES users(id), NOT NULL |
| content | VARCHAR(280) | NOT NULL |
| created_at | TIMESTAMP | DEFAULT NOW() |

### likes
| Column | Type | Constraints |
|--------|------|-------------|
| id | SERIAL | PRIMARY KEY |
| user_id | INTEGER | REFERENCES users(id), NOT NULL |
| tweet_id | INTEGER | REFERENCES tweets(id) ON DELETE CASCADE, NOT NULL |
| created_at | TIMESTAMP | DEFAULT NOW() |
| | | UNIQUE(user_id, tweet_id) |

### follows
| Column | Type | Constraints |
|--------|------|-------------|
| id | SERIAL | PRIMARY KEY |
| follower_id | INTEGER | REFERENCES users(id), NOT NULL |
| following_id | INTEGER | REFERENCES users(id), NOT NULL |
| created_at | TIMESTAMP | DEFAULT NOW() |
| | | UNIQUE(follower_id, following_id) |
| | | CHECK(follower_id != following_id) |

## Integration Points

- Frontend authenticates via `/api/auth/login` and stores JWT in localStorage
- All authenticated requests include `Authorization: Bearer <token>` header
- Backend validates JWT on protected routes using middleware
- Backend connects to PostgreSQL via connection pool (SQLx)
- Frontend uses TanStack Query to cache and invalidate server data
- CORS configured on backend to allow frontend origin (http://localhost:5173)
- Backend runs on port 8080, frontend on port 5173, PostgreSQL on port 5432
- Default credentials: user=admin, password=admin123 (created via seed data)
