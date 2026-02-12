# Twitter-Like App - Design Document

## Architecture Overview

A simple Twitter-like web application with a Rust backend (Axum), React frontend (Vite + TypeScript + Tailwind CSS), and SQLite database. Users can create posts (tweets), like posts, follow other users, and view a timeline feed.

### System Components
- **Backend**: Rust with Axum 0.8 web framework, SQLx 0.8 for database access, SQLite
- **Frontend**: React 19 with TypeScript, Vite, Tailwind CSS, TanStack Router, TanStack Query
- **Database**: SQLite (file-based, no external server needed)

## Backend API Endpoints

### Auth
- `POST /api/auth/register` - Register a new user (username, email, password). Returns JWT token and user profile.
- `POST /api/auth/login` - Login with email/password and receive JWT token.
- `GET /api/auth/me` - Get current authenticated user profile (protected).

### Users
- `GET /api/users/{id}` - Get user profile with followers_count, following_count, is_following (protected)
- `GET /api/users/{id}/followers` - Get list of user's followers (protected)
- `GET /api/users/{id}/following` - Get list of users being followed (protected)
- `POST /api/users/{id}/follow` - Follow a user (protected). Prevents self-follow.
- `DELETE /api/users/{id}/follow` - Unfollow a user (protected)

### Posts
- `GET /api/posts` - Get timeline feed: posts from followed users and own posts, ordered by created_at DESC, limit 50 (protected)
- `POST /api/posts` - Create a new post, max 280 bytes (protected)
- `GET /api/posts/{id}` - Get a single post with enriched data (protected)
- `DELETE /api/posts/{id}` - Delete own post. Deletes associated likes first. Only post author can delete (protected).
- `POST /api/posts/{id}/like` - Like a post. Returns 409 if already liked (protected).
- `DELETE /api/posts/{id}/like` - Unlike a post (protected)
- `GET /api/users/{id}/posts` - Get all posts by a user, ordered by created_at DESC (protected)

### Backend Responsibilities
- JWT-based authentication with 24h token expiry and auth middleware on protected routes
- Input validation (post length max 280 bytes, email format requires @ and ., non-empty fields)
- CORS configuration allowing http://localhost:5173 origin with any methods and headers
- Database connection pooling with SQLx
- Password hashing with Argon2 and random salt per user
- Error handling with proper HTTP status codes (400, 401, 404, 409, 500)
- Post enrichment: each post response includes username, likes_count, and liked_by_me
- Tracing-based logging via tracing-subscriber

## Frontend

### Pages
- **LoginPage** (`/login`) - Login form with email/password fields, error display, link to register
- **RegisterPage** (`/register`) - Registration form with username/email/password fields, error display, link to login
- **HomePage** (`/`) - Timeline feed with PostComposer and Timeline components (auth-protected)
- **ProfilePage** (`/profile/$userId`) - User profile with avatar, username, email, follower/following counts, FollowButton (hidden for own profile), and user's posts (auth-protected)
- **PostDetailPage** (`/post/$postId`) - Single post view using PostCard (auth-protected)

### Components
- **PostCard** - Displays a single post with author avatar (first letter), username (links to profile), relative timestamp, content (links to post detail), and like button with heart icon and count
- **PostComposer** - Textarea with 280-character limit, character counter with color coding (gray/yellow/red), and disabled submit during pending state
- **Navbar** - Sticky top navigation bar with app title (link to home), Home link, Profile link, and Logout button
- **Timeline** - Renders a list of PostCards with loading spinner and empty state messaging
- **FollowButton** - Toggle button between Follow (blue) and Following (outlined) states with hover-to-red-for-unfollow effect

### Interactions
- Click like button toggles like state with optimistic update via TanStack Query (timeline query updated immediately, rollback on error)
- Submit post clears the composer and invalidates the timeline query
- Follow/unfollow invalidates profile, timeline, followers, and following queries
- Navigation between pages via TanStack Router
- Auth guard redirects unauthenticated users to /login via TanStack Router beforeLoad

### Frontend Libraries
- React 19 with TypeScript
- Vite (dev server with API proxy to backend)
- Tailwind CSS
- TanStack Router (type-safe routing with auth guards)
- TanStack Query (data fetching with caching, optimistic updates, query invalidation)

## Database Schema

### Tables
- **users** - id (TEXT PK), username (TEXT UNIQUE NOT NULL), email (TEXT UNIQUE NOT NULL), password_hash (TEXT NOT NULL), created_at (TEXT NOT NULL)
- **posts** - id (TEXT PK), user_id (TEXT NOT NULL FK->users.id), content (TEXT NOT NULL, CHECK length<=280), created_at (TEXT NOT NULL)
- **likes** - PRIMARY KEY (user_id, post_id), user_id (TEXT NOT NULL FK->users.id), post_id (TEXT NOT NULL FK->posts.id), created_at (TEXT NOT NULL)
- **follows** - PRIMARY KEY (follower_id, following_id), follower_id (TEXT NOT NULL FK->users.id), following_id (TEXT NOT NULL FK->users.id), created_at (TEXT NOT NULL), CHECK(follower_id != following_id)

### Indexes
- posts: idx_posts_user_id on user_id, idx_posts_created_at on created_at DESC
- likes: idx_likes_post_id on post_id
- follows: idx_follows_following_id on following_id

### Notes
- All IDs are UUIDs stored as TEXT
- Timestamps are RFC3339 strings set by the backend
- Tables are created automatically by the backend on startup (db.rs create_tables)
- Foreign keys are defined but SQLite foreign key enforcement (PRAGMA foreign_keys) is not enabled at runtime by the backend

## Integration Points
- Frontend calls backend REST API via fetch with JWT in Authorization header (Bearer scheme)
- API client uses relative `/api` prefix, proxied by Vite dev server to http://localhost:3000
- Backend connects to SQLite via SQLx connection pool, database file: `twitter.db`
- Frontend runs on port 5173 (Vite dev), backend on port 3000
- Backend serves CORS headers allowing http://localhost:5173 origin
- Auth tokens stored in localStorage (keys: `token`, `user`)
- TanStack Query stale time: 30 seconds, retry: 1
