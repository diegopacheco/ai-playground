# Feature Documentation - 2026-02-13

## Authentication

### Register
- Endpoint: `POST /api/auth/register`
- Body: `{ username, email, password }`
- Creates new user with BCrypt hashed password
- Returns JWT token and user profile
- Duplicate username/email returns 409 Conflict

### Login
- Endpoint: `POST /api/auth/login`
- Body: `{ email, password }`
- Validates credentials against stored BCrypt hash
- Returns JWT token and user profile
- Invalid credentials return 401 Unauthorized

## User Profiles

### Get User
- Endpoint: `GET /api/users/:id`
- Returns user profile (id, username, email, display_name, bio, created_at)
- Password hash is never included in responses

### Update Profile
- Endpoint: `PUT /api/users/:id` (auth required)
- Body: `{ display_name?, bio? }`
- Only authenticated users can update their own profile

### Followers/Following Lists
- `GET /api/users/:id/followers` - Returns list of users following this user
- `GET /api/users/:id/following` - Returns list of users this user follows

## Tweets

### Create Tweet
- Endpoint: `POST /api/tweets` (auth required)
- Body: `{ content }` (max 280 characters)
- Returns created tweet with author info and like count

### Get Tweet
- Endpoint: `GET /api/tweets/:id`
- Returns tweet with author info and like count

### Delete Tweet
- Endpoint: `DELETE /api/tweets/:id` (auth required)
- Only the tweet owner can delete their tweets
- Returns 204 No Content on success

### Home Feed
- Endpoint: `GET /api/tweets/feed` (auth required)
- Returns tweets from followed users and own tweets
- Ordered by most recent first, limited to 50 tweets

### User Tweets
- Endpoint: `GET /api/users/:id/tweets`
- Returns all tweets by a specific user, ordered by most recent

## Likes

### Like Tweet
- Endpoint: `POST /api/tweets/:id/like` (auth required)
- Idempotent (liking an already-liked tweet is a no-op)

### Unlike Tweet
- Endpoint: `DELETE /api/tweets/:id/like` (auth required)
- Removes the like from the tweet

## Follows

### Follow User
- Endpoint: `POST /api/users/:id/follow` (auth required)
- Cannot follow yourself (returns 400)
- Duplicate follows are handled gracefully

### Unfollow User
- Endpoint: `DELETE /api/users/:id/follow` (auth required)
- Removes the follow relationship

## Frontend Pages

### Login Page (`/login`)
- Email and password fields
- Link to register page
- Redirects to home on success

### Register Page (`/register`)
- Username, email, and password fields
- Link to login page
- Redirects to home on success

### Home Page (`/`)
- Compose box with 280-character counter
- Feed of tweets from followed users and own tweets
- Each tweet shows author, content, timestamp, like count, like/delete actions

### Profile Page (`/profile/:id`)
- User card with display name, username, bio, follower/following counts
- Follow/unfollow button
- List of user's tweets

### Tweet Detail Page (`/tweet/:id`)
- Single tweet view with full details

## Database Schema
- **users**: id, username, email, password_hash, display_name, bio, created_at
- **tweets**: id, user_id (FK), content, created_at
- **likes**: id, user_id (FK), tweet_id (FK with CASCADE), created_at, UNIQUE(user_id, tweet_id)
- **follows**: id, follower_id (FK), following_id (FK), created_at, UNIQUE pair, CHECK no self-follow
