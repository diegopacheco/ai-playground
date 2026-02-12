# Twitter-Like App - Design Document

## Architecture Overview

A simple Twitter-like web application with a Rust backend (Axum), React frontend (Vite + TypeScript + Tailwind CSS), and SQLite database. Users can create posts (tweets), like posts, follow other users, and view a timeline feed.

### System Components
- **Backend**: Rust with Axum web framework, SQLx for database access, SQLite
- **Frontend**: React 19 with TypeScript, Vite, Tailwind CSS, TanStack Query
- **Database**: SQLite (file-based, no external server needed)

## Backend API Endpoints

### Auth
- `POST /api/auth/register` - Register a new user (username, email, password)
- `POST /api/auth/login` - Login and receive JWT token
- `GET /api/auth/me` - Get current user profile

### Users
- `GET /api/users/:id` - Get user profile
- `GET /api/users/:id/followers` - Get user followers
- `GET /api/users/:id/following` - Get users being followed
- `POST /api/users/:id/follow` - Follow a user
- `DELETE /api/users/:id/follow` - Unfollow a user

### Posts
- `GET /api/posts` - Get timeline feed (posts from followed users)
- `POST /api/posts` - Create a new post (max 280 chars)
- `GET /api/posts/:id` - Get a single post
- `DELETE /api/posts/:id` - Delete own post
- `POST /api/posts/:id/like` - Like a post
- `DELETE /api/posts/:id/like` - Unlike a post
- `GET /api/users/:id/posts` - Get posts by a user

### Backend Responsibilities
- JWT-based authentication
- Input validation (post length, email format)
- CORS configuration for frontend
- Database connection pooling with SQLx
- Password hashing with argon2
- Error handling with proper HTTP status codes

## Frontend Components

### Pages
- **LoginPage** - Login form with email/password
- **RegisterPage** - Registration form
- **HomePage** - Timeline feed showing posts from followed users
- **ProfilePage** - User profile with their posts, follower/following counts
- **PostDetailPage** - Single post view

### Components
- **PostCard** - Displays a single post with author, content, like button, timestamp
- **PostComposer** - Text input for creating new posts with character counter
- **UserCard** - User info with follow/unfollow button
- **Navbar** - Navigation bar with links and logout
- **Timeline** - Scrollable list of PostCards
- **FollowButton** - Toggle follow/unfollow state

### Interactions
- Click like button toggles like state (optimistic update via TanStack Query)
- Submit post refreshes timeline
- Follow/unfollow updates follower counts
- Navigation between pages via React Router

## Database Schema

### Tables
- **users** - id, username, email, password_hash, created_at
- **posts** - id, user_id (FK), content, created_at
- **likes** - id, user_id (FK), post_id (FK), created_at (unique user_id+post_id)
- **follows** - id, follower_id (FK), following_id (FK), created_at (unique follower+following)

### Indexes
- users: unique on username, unique on email
- posts: index on user_id, index on created_at DESC
- likes: unique on (user_id, post_id), index on post_id
- follows: unique on (follower_id, following_id), index on following_id

## Integration Points
- Frontend calls backend REST API via fetch with JWT in Authorization header
- Backend connects to SQLite via SQLx connection pool
- Frontend runs on port 5173 (Vite dev), backend on port 3000
- Backend serves CORS headers allowing frontend origin
