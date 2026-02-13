# Feature Documentation - Twitter-like App

**Date**: 2026-02-12

## API Endpoints

### Users

#### POST /api/users - Register
Request:
```json
{
  "username": "alice",
  "email": "alice@mail.com",
  "password": "secret123"
}
```
Response (201):
```json
{
  "token": "eyJ...",
  "user": {
    "id": 1,
    "username": "alice",
    "email": "alice@mail.com",
    "created_at": "2026-02-12 10:00:00"
  }
}
```

#### POST /api/users/login - Login
Request:
```json
{
  "username": "alice",
  "password": "secret123"
}
```
Response (200):
```json
{
  "token": "eyJ...",
  "user": {
    "id": 1,
    "username": "alice",
    "email": "alice@mail.com",
    "created_at": "2026-02-12 10:00:00"
  }
}
```

#### GET /api/users/me - Get Current User (Authenticated)
Response (200):
```json
{
  "id": 1,
  "username": "alice",
  "email": "alice@mail.com",
  "created_at": "2026-02-12 10:00:00"
}
```

#### GET /api/users/{id} - Get User Profile
Response (200):
```json
{
  "id": 1,
  "username": "alice",
  "email": "alice@mail.com",
  "created_at": "2026-02-12 10:00:00"
}
```

#### GET /api/users/{id}/followers - List Followers
Response (200):
```json
[
  {
    "id": 2,
    "username": "bob",
    "email": "bob@mail.com",
    "created_at": "2026-02-12 10:00:00"
  }
]
```

#### GET /api/users/{id}/following - List Following
Response (200): Same format as followers.

### Posts

#### POST /api/posts - Create Post (Authenticated)
Request:
```json
{
  "content": "Hello world!"
}
```
Response (200):
```json
{
  "id": 1,
  "user_id": 1,
  "content": "Hello world!",
  "created_at": "2026-02-12 10:05:00",
  "username": "alice",
  "like_count": 0
}
```

#### GET /api/posts - Get All Posts
Query parameters: `page` (default 1), `limit` (default 20, max 100)

Response (200):
```json
[
  {
    "id": 1,
    "user_id": 1,
    "content": "Hello world!",
    "created_at": "2026-02-12 10:05:00",
    "username": "alice",
    "like_count": 3
  }
]
```

#### GET /api/posts/{id} - Get Single Post
Response (200): Same format as a single post object above.

#### DELETE /api/posts/{id} - Delete Post (Authenticated, Owner Only)
Response (200):
```json
{
  "message": "Post deleted"
}
```

### Feed

#### GET /api/feed - Get Personalized Feed (Authenticated)
Returns posts from followed users and the authenticated user's own posts.
Query parameters: `page` (default 1), `limit` (default 20, max 100)

Response (200): Same format as Get All Posts.

### Likes

#### POST /api/posts/{id}/like - Like a Post (Authenticated)
Response (200):
```json
{
  "message": "Post liked"
}
```

#### DELETE /api/posts/{id}/like - Unlike a Post (Authenticated)
Response (200):
```json
{
  "message": "Post unliked"
}
```

#### GET /api/posts/{id}/likes - Get Like Count
Response (200):
```json
{
  "post_id": 1,
  "count": 5
}
```

### Follows

#### POST /api/users/{id}/follow - Follow a User (Authenticated)
Response (200):
```json
{
  "message": "User followed"
}
```

#### DELETE /api/users/{id}/follow - Unfollow a User (Authenticated)
Response (200):
```json
{
  "message": "User unfollowed"
}
```

## Frontend Pages

### LoginPage
- Username and password form fields
- Submit button with loading state ("Logging in...")
- Error display for failed login
- Link to navigate to RegisterPage

### RegisterPage
- Username, email, and password form fields
- Password field has `minLength={6}` HTML validation
- Submit button with loading state ("Creating account...")
- Error display for failed registration
- Link to navigate to LoginPage
- On success, auto-logs in and navigates to home

### HomePage
- PostComposer at the top for creating new posts
- Feed component showing personalized feed (posts from followed users + own posts)
- Infinite scroll pagination via "Load More" button
- Clicking a post author navigates to their profile
- Clicking post content navigates to post detail

### ProfilePage
- User avatar (first letter of username), username, join date
- "You" badge for own profile
- Three tabs: Posts, Followers, Following
- Posts tab shows the user's posts in a feed
- Followers/Following tabs show UserCard lists with follow/unfollow buttons

### PostDetailPage
- Full post view with author info, timestamp, content
- Like/Unlike button with count
- Back to feed navigation link
- Clicking author navigates to their profile

## Frontend Components

### Navbar
- App title "Chirp" (links to home)
- Home and Profile navigation buttons
- Username display with @ prefix
- Logout button

### PostCard
- Author avatar, username, relative timestamp
- Post content
- Like/Unlike toggle button with heart icon and count
- Click handlers for author and post navigation

### PostComposer
- Textarea with "What's happening?" placeholder
- Character counter (280 max) with color coding (gray > yellow > red)
- Post button disabled when empty or over limit
- Clears on successful post and invalidates feed/posts queries

### UserCard
- User avatar and username
- Follow/Unfollow toggle button (hidden for own profile)
- Click handler for user navigation

### FollowersList
- Fetches followers or following list for a given user
- Renders UserCard for each user
- Loading spinner and error/empty states

### Feed
- Accepts queryKey and fetchFn props for flexible data source
- Uses TanStack Query `useInfiniteQuery` for pagination
- "Load More" button for next page
- Loading spinner, error state, and empty state

## Authentication Flow

1. User enters credentials on LoginPage (or RegisterPage)
2. Frontend calls `POST /api/users/login` (or `POST /api/users`)
3. Backend validates credentials and returns JWT token + user object
4. Frontend stores token and user in React Context and localStorage
5. Subsequent API calls include `Authorization: Bearer <token>` header via the `apiClient` function
6. On logout, token and user are cleared from Context and localStorage
7. Token expires after 24 hours; user must log in again

## User Interactions

### Posting
- Type content in PostComposer, click Post
- Content is validated (non-empty, max 280 chars)
- On success, feed and posts queries are invalidated to show new post

### Liking
- Click heart icon on PostCard or PostDetailPage
- Sends POST or DELETE to like/unlike endpoint
- Invalidates related queries to update UI

### Following
- Click Follow/Unfollow button on UserCard
- Sends POST or DELETE to follow endpoint
- Invalidates followers/following queries

### Feed
- Home page shows personalized feed (own posts + followed users' posts)
- Paginated with "Load More" button
- Posts sorted by newest first
