# Feature Documentation - 2026-02-12

## API Endpoints

### Authentication

#### POST /api/auth/register
Register a new user account.

**Request Body**:
```json
{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

**Response 200**:
```json
{
  "token": "jwt-string",
  "user": {
    "id": "uuid-string",
    "username": "string",
    "email": "string",
    "created_at": "rfc3339-timestamp"
  }
}
```

**Error Responses**:
- 400: All fields are required / Invalid email format
- 409: Username or email already exists

**Validation**:
- username, email, password must be non-empty
- email must contain `@` and `.`
- username and email must be unique

---

#### POST /api/auth/login
Authenticate and receive a JWT token.

**Request Body**:
```json
{
  "email": "string",
  "password": "string"
}
```

**Response 200**:
```json
{
  "token": "jwt-string",
  "user": {
    "id": "uuid-string",
    "username": "string",
    "email": "string",
    "created_at": "rfc3339-timestamp"
  }
}
```

**Error Responses**:
- 401: Invalid credentials

---

#### GET /api/auth/me
Get the currently authenticated user's profile.

**Headers**: `Authorization: Bearer <token>`

**Response 200**:
```json
{
  "id": "uuid-string",
  "username": "string",
  "email": "string",
  "created_at": "rfc3339-timestamp"
}
```

**Error Responses**:
- 401: Missing authorization header / Invalid token
- 404: User not found

---

### Users

#### GET /api/users/:id
Get a user's profile with follower/following counts and follow status.

**Headers**: `Authorization: Bearer <token>`

**Response 200**:
```json
{
  "user": {
    "id": "uuid-string",
    "username": "string",
    "email": "string",
    "created_at": "rfc3339-timestamp"
  },
  "followers_count": 0,
  "following_count": 0,
  "is_following": false
}
```

**Error Responses**:
- 401: Unauthorized
- 404: User not found

---

#### GET /api/users/:id/followers
Get the list of users following the specified user.

**Headers**: `Authorization: Bearer <token>`

**Response 200**:
```json
[
  {
    "id": "uuid-string",
    "username": "string",
    "email": "string",
    "created_at": "rfc3339-timestamp"
  }
]
```

---

#### GET /api/users/:id/following
Get the list of users the specified user is following.

**Headers**: `Authorization: Bearer <token>`

**Response 200**:
```json
[
  {
    "id": "uuid-string",
    "username": "string",
    "email": "string",
    "created_at": "rfc3339-timestamp"
  }
]
```

---

#### POST /api/users/:id/follow
Follow a user.

**Headers**: `Authorization: Bearer <token>`

**Response 200**:
```json
{
  "message": "Followed successfully"
}
```

**Error Responses**:
- 400: Cannot follow yourself
- 404: User not found
- 409: Already following this user

---

#### DELETE /api/users/:id/follow
Unfollow a user.

**Headers**: `Authorization: Bearer <token>`

**Response 200**:
```json
{
  "message": "Unfollowed successfully"
}
```

**Error Responses**:
- 404: Not following this user

---

### Posts

#### GET /api/posts
Get the timeline feed. Returns posts from followed users and the authenticated user's own posts, ordered by newest first, limited to 50.

**Headers**: `Authorization: Bearer <token>`

**Response 200**:
```json
[
  {
    "id": "uuid-string",
    "user_id": "uuid-string",
    "username": "string",
    "content": "string",
    "likes_count": 0,
    "liked_by_me": false,
    "created_at": "rfc3339-timestamp"
  }
]
```

---

#### POST /api/posts
Create a new post.

**Headers**: `Authorization: Bearer <token>`

**Request Body**:
```json
{
  "content": "string (max 280 characters)"
}
```

**Response 200**:
```json
{
  "id": "uuid-string",
  "user_id": "uuid-string",
  "username": "string",
  "content": "string",
  "likes_count": 0,
  "liked_by_me": false,
  "created_at": "rfc3339-timestamp"
}
```

**Error Responses**:
- 400: Post content cannot be empty / Post content cannot exceed 280 characters

---

#### GET /api/posts/:id
Get a single post by ID.

**Headers**: `Authorization: Bearer <token>`

**Response 200**: Same as PostResponse format above.

**Error Responses**:
- 404: Post not found

---

#### DELETE /api/posts/:id
Delete a post. Only the post author can delete it.

**Headers**: `Authorization: Bearer <token>`

**Response 200**:
```json
{
  "message": "Post deleted"
}
```

**Error Responses**:
- 401: Cannot delete another user's post
- 404: Post not found

---

#### POST /api/posts/:id/like
Like a post.

**Headers**: `Authorization: Bearer <token>`

**Response 200**:
```json
{
  "message": "Post liked"
}
```

**Error Responses**:
- 404: Post not found
- 409: Already liked this post

---

#### DELETE /api/posts/:id/like
Unlike a post.

**Headers**: `Authorization: Bearer <token>`

**Response 200**:
```json
{
  "message": "Post unliked"
}
```

**Error Responses**:
- 404: Like not found

---

#### GET /api/users/:id/posts
Get all posts by a specific user, ordered by newest first.

**Headers**: `Authorization: Bearer <token>`

**Response 200**: Array of PostResponse objects.

---

## Frontend Pages

### LoginPage (`/login`)
Login form with email and password fields. Shows error messages on failed login. Links to registration page. Redirects to home on successful authentication.

### RegisterPage (`/register`)
Registration form with username, email, and password fields. Shows error messages on failed registration. Links to login page. Redirects to home on successful registration.

### HomePage (`/`)
Protected route. Displays the PostComposer for creating new posts and the Timeline showing posts from followed users and the current user. Shows a loading spinner while data is being fetched.

### ProfilePage (`/profile/$userId`)
Protected route. Shows user avatar (first letter of username), username, email, follower/following counts, and a Follow/Unfollow button (hidden when viewing own profile). Displays the user's posts below the profile header.

### PostDetailPage (`/post/$postId`)
Protected route. Displays a single post using the PostCard component. Shows loading spinner or "Post not found" message as appropriate.

---

## Frontend Components

### Navbar
Sticky top navigation bar with the app title (link to home), Home link, Profile link (current user), and Logout button.

### PostComposer
Textarea with 280-character limit and character counter. Counter changes color: gray (normal), yellow (under 20 remaining), red (over limit). Submit button is disabled when content is empty, over limit, or request is pending.

### PostCard
Displays a post with the author avatar (first letter), username (links to profile), relative timestamp, post content (links to post detail), and a like button with count. Like button shows filled/outlined heart icon based on like state.

### Timeline
Renders a list of PostCard components. Shows a loading spinner during data fetch and a "No posts yet" message when the list is empty.

### FollowButton
Toggle button that switches between "Follow" (blue) and "Following" (outlined with red hover) states. Disabled while the follow/unfollow request is pending.

---

## Database Schema

### users
| Column | Type | Constraints |
|---|---|---|
| id | TEXT | PRIMARY KEY |
| username | TEXT | NOT NULL, UNIQUE |
| email | TEXT | NOT NULL, UNIQUE |
| password_hash | TEXT | NOT NULL |
| created_at | TEXT | NOT NULL |

### posts
| Column | Type | Constraints |
|---|---|---|
| id | TEXT | PRIMARY KEY |
| user_id | TEXT | NOT NULL, FK -> users(id) |
| content | TEXT | NOT NULL, CHECK(length <= 280) |
| created_at | TEXT | NOT NULL |

### likes
| Column | Type | Constraints |
|---|---|---|
| user_id | TEXT | NOT NULL, FK -> users(id) |
| post_id | TEXT | NOT NULL, FK -> posts(id) |
| created_at | TEXT | NOT NULL |
| | | PRIMARY KEY (user_id, post_id) |

### follows
| Column | Type | Constraints |
|---|---|---|
| follower_id | TEXT | NOT NULL, FK -> users(id) |
| following_id | TEXT | NOT NULL, FK -> users(id) |
| created_at | TEXT | NOT NULL |
| | | PRIMARY KEY (follower_id, following_id) |
| | | CHECK(follower_id != following_id) |

### Indexes
- `idx_posts_user_id` on posts(user_id)
- `idx_posts_created_at` on posts(created_at DESC)
- `idx_likes_post_id` on likes(post_id)
- `idx_follows_following_id` on follows(following_id)

---

## Configuration

### Backend
- Listens on `0.0.0.0:3000`
- SQLite database file: `twitter.db` (created automatically)
- JWT expiration: 24 hours
- CORS origin: `http://localhost:5173`
- Tables created automatically on startup

### Frontend
- Vite dev server on port 5173
- API proxy configured to forward `/api` to `http://localhost:3000`
- TanStack Query stale time: 30 seconds
- TanStack Query retry: 1
- Auth token stored in localStorage under key `token`
- User data stored in localStorage under key `user`
