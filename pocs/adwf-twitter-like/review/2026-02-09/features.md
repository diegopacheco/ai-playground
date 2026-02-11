# Twitter Clone - Feature Documentation

## Overview

The Twitter Clone is a full-stack social media application that replicates core Twitter functionality. Built with a Rust backend (Axum framework) and React frontend (TypeScript, Vite), the application provides a modern, high-performance platform for social networking.

### Technology Stack

**Backend:**
- Rust 1.93+ (Edition 2024)
- Axum 0.8 web framework
- Tokio async runtime
- SQLx 0.8 with PostgreSQL
- JWT authentication
- bcrypt password hashing

**Frontend:**
- React 19
- TypeScript
- Vite build tool
- TanStack Query (React Query)
- React Router DOM
- Tailwind CSS

**Database:**
- PostgreSQL 18
- Automated migrations
- Connection pooling
- Optimized indexes

### Architecture

The application follows a modular, layered architecture:
- **Models Layer**: Data structures and validation
- **Handlers Layer**: Business logic and API endpoints
- **Routes Layer**: HTTP routing and middleware
- **Middleware Layer**: Authentication and request processing
- **Frontend Layer**: Component-based UI with React

## Feature List

### Core Features

1. **User Authentication**: Register, login, logout with JWT tokens
2. **User Profiles**: View and update profiles with statistics
3. **Tweet Management**: Create, view, and delete tweets (280 character limit)
4. **Social Graph**: Follow and unfollow users
5. **Feed Generation**: Chronological feed of tweets from followed users
6. **Social Interactions**: Like, retweet, and comment on tweets
7. **User Discovery**: View followers and following lists
8. **Real-time Updates**: Optimistic UI updates with React Query
9. **Responsive Design**: Mobile-first design with Tailwind CSS
10. **Performance**: Connection pooling, indexes, and async operations

## User Authentication

### Registration

Create a new user account with username, email, and password.

**Validation Rules:**
- Username: 3-50 characters, unique
- Email: Valid email format, unique
- Password: Minimum 6 characters
- Password hashing: bcrypt with cost factor 12

**API Endpoint:**
```bash
POST /api/auth/register
Content-Type: application/json

{
  "username": "alice",
  "email": "alice@test.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": 1,
    "username": "alice",
    "email": "alice@test.com",
    "display_name": null,
    "bio": null,
    "created_at": "2026-02-09T12:00:00Z",
    "updated_at": "2026-02-09T12:00:00Z"
  }
}
```

**Frontend Usage:**
```typescript
import { authApi } from '@/lib/api';

const handleRegister = async () => {
  const response = await authApi.register({
    username: 'alice',
    email: 'alice@test.com',
    password: 'password123'
  });
  localStorage.setItem('token', response.token);
};
```

### Login

Authenticate existing users and receive JWT token.

**API Endpoint:**
```bash
POST /api/auth/login
Content-Type: application/json

{
  "username": "alice",
  "password": "password123"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": 1,
    "username": "alice",
    "email": "alice@test.com"
  }
}
```

**Token Specification:**
- Algorithm: HS256
- Expiration: 7 days
- Claims: user_id, exp (expiration)
- Storage: localStorage (frontend)

### Logout

Clear authentication state and invalidate session.

**API Endpoint:**
```bash
POST /api/auth/logout
Authorization: Bearer <token>
```

**Response:** 204 No Content

**Frontend Usage:**
```typescript
const handleLogout = async () => {
  await authApi.logout();
  localStorage.removeItem('token');
  navigate('/login');
};
```

### Protected Routes

All endpoints except `/api/auth/register` and `/api/auth/login` require authentication.

**Authentication Middleware:**
- Extracts JWT token from Authorization header
- Validates token signature and expiration
- Extracts user ID from claims
- Attaches user ID to request context
- Returns 401 Unauthorized for invalid/missing tokens

## User Profiles

### View Profile

Get detailed user profile with statistics.

**API Endpoint:**
```bash
GET /api/users/:id
Authorization: Bearer <token>
```

**Response:**
```json
{
  "id": 1,
  "username": "alice",
  "email": "alice@test.com",
  "display_name": "Alice Smith",
  "bio": "Software developer",
  "created_at": "2026-02-09T12:00:00Z",
  "updated_at": "2026-02-09T12:00:00Z",
  "followers_count": 10,
  "following_count": 15,
  "tweets_count": 42
}
```

**Statistics:**
- **followers_count**: Number of users following this user
- **following_count**: Number of users this user follows
- **tweets_count**: Total tweets created by user

**Frontend Component:**
```typescript
import { usersApi } from '@/lib/api';
import { useQuery } from '@tanstack/react-query';

const ProfilePage = () => {
  const { data: user } = useQuery({
    queryKey: ['user', userId],
    queryFn: () => usersApi.getUser(userId)
  });

  return (
    <div>
      <h1>{user.display_name || user.username}</h1>
      <p>{user.bio}</p>
      <div>
        <span>{user.followers_count} Followers</span>
        <span>{user.following_count} Following</span>
        <span>{user.tweets_count} Tweets</span>
      </div>
    </div>
  );
};
```

### Update Profile

Modify display name and bio.

**API Endpoint:**
```bash
PUT /api/users/:id
Authorization: Bearer <token>
Content-Type: application/json

{
  "display_name": "Alice Smith",
  "bio": "Software developer"
}
```

**Validation:**
- display_name: Max 100 characters, optional
- bio: Max 500 characters, optional
- Authorization: Only the profile owner can update

**Response:**
```json
{
  "id": 1,
  "username": "alice",
  "display_name": "Alice Smith",
  "bio": "Software developer"
}
```

## Following System

### Follow User

Create a follow relationship.

**API Endpoint:**
```bash
POST /api/users/:id/follow
Authorization: Bearer <token>
```

**Response:** 201 Created

**Database Constraints:**
- Composite primary key: (follower_id, following_id)
- Check constraint prevents self-following
- CASCADE deletion when user deleted
- Bidirectional indexes for fast lookups

**Effect:**
- Increases target user's followers_count
- Increases current user's following_count
- Followed user's tweets appear in feed

### Unfollow User

Remove follow relationship.

**API Endpoint:**
```bash
DELETE /api/users/:id/follow
Authorization: Bearer <token>
```

**Response:** 204 No Content

**Effect:**
- Decreases target user's followers_count
- Decreases current user's following_count
- Followed user's tweets removed from feed

### Get Followers

List users following a specific user.

**API Endpoint:**
```bash
GET /api/users/:id/followers
Authorization: Bearer <token>
```

**Response:**
```json
[
  {
    "id": 2,
    "username": "bob",
    "display_name": "Bob Jones",
    "bio": "Designer"
  }
]
```

### Get Following

List users that a specific user follows.

**API Endpoint:**
```bash
GET /api/users/:id/following
Authorization: Bearer <token>
```

**Response:**
```json
[
  {
    "id": 3,
    "username": "charlie",
    "display_name": "Charlie Brown"
  }
]
```

## Tweet Operations

### Create Tweet

Post a new tweet with content validation.

**API Endpoint:**
```bash
POST /api/tweets
Authorization: Bearer <token>
Content-Type: application/json

{
  "content": "Hello world!"
}
```

**Validation:**
- Content: 1-280 characters
- Required field
- Whitespace trimmed

**Response:**
```json
{
  "id": 1,
  "user_id": 1,
  "content": "Hello world!",
  "created_at": "2026-02-09T12:00:00Z",
  "updated_at": "2026-02-09T12:00:00Z",
  "author_username": "alice",
  "author_display_name": "Alice Smith",
  "likes_count": 0,
  "retweets_count": 0,
  "comments_count": 0,
  "is_liked": false,
  "is_retweeted": false
}
```

**Frontend Component:**
```typescript
const TweetComposer = () => {
  const [content, setContent] = useState('');
  const createMutation = useMutation({
    mutationFn: (content: string) => tweetsApi.createTweet(content),
    onSuccess: () => {
      queryClient.invalidateQueries(['feed']);
      setContent('');
    }
  });

  return (
    <div>
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        maxLength={280}
        placeholder="What's happening?"
      />
      <div>{content.length}/280</div>
      <button
        onClick={() => createMutation.mutate(content)}
        disabled={!content.trim() || content.length > 280}
      >
        Tweet
      </button>
    </div>
  );
};
```

### View Tweet

Get single tweet with metadata.

**API Endpoint:**
```bash
GET /api/tweets/:id
Authorization: Bearer <token>
```

**Response Includes:**
- Tweet content and timestamps
- Author username and display name
- Counts: likes, retweets, comments
- User-specific flags: is_liked, is_retweeted

### Delete Tweet

Remove a tweet (author only).

**API Endpoint:**
```bash
DELETE /api/tweets/:id
Authorization: Bearer <token>
```

**Response:** 204 No Content

**Authorization:**
- Only the tweet author can delete
- Returns 403 Forbidden for non-authors

**Cascade Effects:**
- Deletes all likes on the tweet
- Deletes all retweets of the tweet
- Deletes all comments on the tweet

### Get User Tweets

Retrieve tweets from a specific user.

**API Endpoint:**
```bash
GET /api/tweets/user/:userId?limit=20&offset=0
Authorization: Bearer <token>
```

**Query Parameters:**
- limit: Number of tweets (default: 20)
- offset: Pagination offset (default: 0)

**Response:**
```json
[
  {
    "id": 1,
    "content": "Hello world!",
    "created_at": "2026-02-09T12:00:00Z",
    "author_username": "alice"
  }
]
```

**Ordering:** Most recent first (created_at DESC)

## Social Interactions

### Like Tweet

Add a like to a tweet.

**API Endpoint:**
```bash
POST /api/tweets/:id/like
Authorization: Bearer <token>
```

**Response:** 201 Created

**Effect:**
- Increments tweet's likes_count
- Sets is_liked = true for user
- Creates like record in database

**Idempotency:**
- Duplicate likes return success (no error)
- Uses composite primary key (user_id, tweet_id)

### Unlike Tweet

Remove a like from a tweet.

**API Endpoint:**
```bash
DELETE /api/tweets/:id/like
Authorization: Bearer <token>
```

**Response:** 204 No Content

**Effect:**
- Decrements tweet's likes_count
- Sets is_liked = false for user
- Removes like record from database

### Retweet

Share a tweet to your followers.

**API Endpoint:**
```bash
POST /api/tweets/:id/retweet
Authorization: Bearer <token>
```

**Response:** 201 Created

**Effect:**
- Increments tweet's retweets_count
- Sets is_retweeted = true for user
- Creates retweet record in database
- Appears in followers' feeds

### Remove Retweet

Remove retweet.

**API Endpoint:**
```bash
DELETE /api/tweets/:id/retweet
Authorization: Bearer <token>
```

**Response:** 204 No Content

**Effect:**
- Decrements tweet's retweets_count
- Sets is_retweeted = false for user

### Add Comment

Post a comment on a tweet.

**API Endpoint:**
```bash
POST /api/tweets/:id/comments
Authorization: Bearer <token>
Content-Type: application/json

{
  "content": "Great tweet!"
}
```

**Validation:**
- Content: 1-280 characters
- Required field

**Response:**
```json
{
  "id": 1,
  "user_id": 1,
  "tweet_id": 1,
  "content": "Great tweet!",
  "created_at": "2026-02-09T12:00:00Z",
  "updated_at": "2026-02-09T12:00:00Z",
  "author_username": "alice",
  "author_display_name": "Alice Smith"
}
```

### Get Comments

Retrieve all comments for a tweet.

**API Endpoint:**
```bash
GET /api/tweets/:id/comments
Authorization: Bearer <token>
```

**Response:**
```json
[
  {
    "id": 1,
    "content": "Great tweet!",
    "author_username": "alice",
    "created_at": "2026-02-09T12:00:00Z"
  }
]
```

**Ordering:** Chronological order (created_at ASC)

### Delete Comment

Remove a comment (author only).

**API Endpoint:**
```bash
DELETE /api/comments/:id
Authorization: Bearer <token>
```

**Response:** 204 No Content

**Authorization:**
- Only comment author can delete
- Returns 403 Forbidden for non-authors

## Feed Generation

### Get Feed

Retrieve personalized feed of tweets from followed users.

**API Endpoint:**
```bash
GET /api/tweets/feed?limit=20&offset=0
Authorization: Bearer <token>
```

**Query Parameters:**
- limit: Number of tweets (default: 20, max: 100)
- offset: Pagination offset (default: 0)

**Response:**
```json
[
  {
    "id": 1,
    "content": "Hello world!",
    "created_at": "2026-02-09T12:00:00Z",
    "author_username": "alice",
    "author_display_name": "Alice Smith",
    "likes_count": 5,
    "retweets_count": 2,
    "comments_count": 3,
    "is_liked": false,
    "is_retweeted": false
  }
]
```

**Feed Algorithm:**
1. Query tweets from users the current user follows
2. Order by created_at DESC (most recent first)
3. Include user's own tweets
4. Apply pagination (limit/offset)
5. Join with user data for author information
6. Calculate interaction counts
7. Set user-specific flags (is_liked, is_retweeted)

**Performance Optimizations:**
- Index on tweets(user_id, created_at)
- Index on follows(follower_id, following_id)
- Connection pooling (5-20 connections)
- Async query execution

**Frontend Implementation:**
```typescript
const HomePage = () => {
  const { data: tweets, isLoading } = useQuery({
    queryKey: ['feed'],
    queryFn: () => tweetsApi.getFeed()
  });

  return (
    <div>
      <TweetComposer />
      {isLoading ? (
        <div>Loading...</div>
      ) : (
        <FeedList tweets={tweets} />
      )}
    </div>
  );
};
```

## Frontend Features

### Components

**NavigationBar**
- Logo and app branding
- Home and profile links
- Logout button
- Current user display
- Responsive mobile menu

**TweetCard**
- Tweet content display
- Author information with avatar
- Timestamp formatting
- Like/retweet/comment buttons
- Interaction counts
- Navigation to tweet detail
- Delete button (own tweets)

**TweetComposer**
- Textarea with character counter
- 280 character limit visualization
- Submit button state management
- Optimistic updates
- Error handling

**CommentList**
- Comment display
- Author information
- Timestamp
- Delete button (own comments)
- Loading states
- Empty state

**FeedList**
- Tweet list rendering
- Loading spinner
- Empty state
- Error boundary

**UserCard**
- User avatar
- Username and display name
- Bio display
- Follow/unfollow button
- Statistics (followers, following, tweets)

### Pages

**LoginPage**
- Toggle between login and signup
- Form validation
- Error display
- Loading states
- Auto-redirect when authenticated

**HomePage**
- Tweet composer at top
- Feed display
- Infinite scroll (optional)
- Real-time updates

**ProfilePage**
- User information display
- Tabs: Tweets, Followers, Following
- Follow/unfollow button
- Profile edit (own profile)
- Tweet list
- User lists

**TweetDetailPage**
- Full tweet display
- Comment list
- Comment composer
- Like/retweet buttons
- Share functionality

### Context Providers

**AuthContext**
- User authentication state
- Login/logout functions
- Current user information
- Token management
- Protected route logic

### State Management

**React Query**
- Server state caching
- Optimistic updates
- Automatic refetching
- Background updates
- Query invalidation
- Mutation handling

**Query Keys:**
- `['feed']` - User feed
- `['tweet', id]` - Single tweet
- `['user', id]` - User profile
- `['userTweets', id]` - User tweets
- `['followers', id]` - User followers
- `['following', id]` - User following
- `['comments', tweetId]` - Tweet comments

### Routing

**React Router DOM**
- `/login` - Authentication page
- `/` - Home feed (protected)
- `/profile/:userId` - User profile (protected)
- `/tweet/:tweetId` - Tweet detail (protected)
- `*` - 404 redirect to home

**Protected Routes:**
- Redirect to `/login` when not authenticated
- Auto-redirect to `/` when authenticated on login page

### Styling

**Tailwind CSS**
- Utility-first CSS framework
- Responsive design breakpoints
- Mobile-first approach
- Custom configuration
- Dark mode support (optional)

**Design System:**
- Primary color: Blue (#1DA1F2)
- Background: White/Gray
- Text: Dark gray/Black
- Borders: Light gray
- Hover states: Darker shades

## Database Features

### Schema Design

**users table:**
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(100),
    bio VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
```

**tweets table:**
```sql
CREATE TABLE tweets (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    content VARCHAR(280) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_tweets_user_id ON tweets(user_id);
CREATE INDEX idx_tweets_created_at ON tweets(created_at);
```

**follows table:**
```sql
CREATE TABLE follows (
    follower_id INTEGER NOT NULL,
    following_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (follower_id, following_id),
    FOREIGN KEY (follower_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (following_id) REFERENCES users(id) ON DELETE CASCADE,
    CHECK (follower_id != following_id)
);

CREATE INDEX idx_follows_follower ON follows(follower_id);
CREATE INDEX idx_follows_following ON follows(following_id);
```

**likes table:**
```sql
CREATE TABLE likes (
    user_id INTEGER NOT NULL,
    tweet_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, tweet_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (tweet_id) REFERENCES tweets(id) ON DELETE CASCADE
);

CREATE INDEX idx_likes_user ON likes(user_id);
CREATE INDEX idx_likes_tweet ON likes(tweet_id);
```

**retweets table:**
```sql
CREATE TABLE retweets (
    user_id INTEGER NOT NULL,
    tweet_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, tweet_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (tweet_id) REFERENCES tweets(id) ON DELETE CASCADE
);

CREATE INDEX idx_retweets_user ON retweets(user_id);
CREATE INDEX idx_retweets_tweet ON retweets(tweet_id);
```

**comments table:**
```sql
CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    tweet_id INTEGER NOT NULL,
    content VARCHAR(280) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (tweet_id) REFERENCES tweets(id) ON DELETE CASCADE
);

CREATE INDEX idx_comments_tweet ON comments(tweet_id);
CREATE INDEX idx_comments_user ON comments(user_id);
CREATE INDEX idx_comments_created_at ON comments(created_at);
```

### Migrations

**Automated Migration System:**
- Migrations run automatically on server startup
- SQLx migration tool
- Version control for schema changes
- Rollback support

**Migration Files:**
- 20260209000001_create_users.sql
- 20260209000002_create_tweets.sql
- 20260209000003_create_follows.sql
- 20260209000004_create_likes.sql
- 20260209000005_create_retweets.sql
- 20260209000006_create_comments.sql

### Connection Pooling

**SQLx PgPool Configuration:**
```rust
let pool = PgPoolOptions::new()
    .max_connections(5)
    .connect(&config.database_url)
    .await?;
```

**Settings:**
- Max connections: 5 (development), 20 (production)
- Automatic connection recycling
- Connection timeout: 30 seconds
- Idle timeout: 10 minutes

## Testing Features

### Integration Tests

**Test Coverage:**
- 13 comprehensive integration tests
- Authentication flow
- User profile operations
- Follow/unfollow operations
- Tweet operations
- Like/unlike operations
- Retweet operations
- Comment operations
- Feed generation
- Authorization checks
- Data persistence
- Complex user interactions

**Running Tests:**
```bash
./run-integration-tests.sh
```

**Test Database:**
- Dedicated PostgreSQL instance on port 5433
- Automatic setup and teardown
- Isolated from production data

### End-to-End Tests

**Playwright Test Suite:**
- 86 end-to-end tests
- Authentication tests (8)
- Tweet tests (11)
- Comment tests (10)
- Profile tests (12)
- Follow/unfollow tests (7)
- Feed tests (12)
- Navigation tests (13)
- Responsive design tests (13)

**Running E2E Tests:**
```bash
./run-e2e-tests.sh
```

**Test Automation:**
- Page Object Model pattern
- Automatic server startup
- Clean test data
- Screenshot on failure
- HTML report generation

### Performance Tests

**k6 Load Testing:**
- 7 comprehensive test suites
- Baseline test (30s, 10 VUs)
- Authentication test (2m, 0→50 VUs)
- User profile test (2.5m, 0→40 VUs)
- Tweet feed test (3m, 0→75 VUs)
- Social interactions test (4.5m, 0→60 VUs)
- Load test (4.5m, 0→50 VUs)
- Stress test (9m, 0→100 VUs)

**Performance Targets:**
- p95 response time: < 500ms
- p99 response time: < 1000ms
- Error rate: < 1%
- Throughput: 50-100 req/s

**Running Performance Tests:**
```bash
./run-k6-tests.sh
```

### Unit Tests

**Cargo Test:**
- Model validation tests
- JWT creation and verification
- Error handling tests
- Request validation tests

**Running Unit Tests:**
```bash
cargo test
```

## Configuration Options

### Environment Variables

**Backend Configuration:**

**DATABASE_URL** (required)
- PostgreSQL connection string
- Format: `postgresql://user:password@host:port/database`
- Example: `postgresql://twitter_user:twitter_pass@localhost:5432/twitter_db`

**JWT_SECRET** (required)
- Secret key for JWT signing
- Minimum 32 characters recommended
- Change in production
- Example: `your-very-secure-secret-key-change-in-production`

**RUST_LOG** (optional)
- Logging level configuration
- Values: error, warn, info, debug, trace
- Default: info
- Example: `twitter_clone=debug,tower_http=debug`

**Frontend Configuration:**

**API_BASE_URL**
- Backend API base URL
- Configured in `src/lib/api.ts`
- Default: `http://localhost:8000/api`

### Server Configuration

**Port:** 8000 (configurable in src/main.rs)
**Host:** 0.0.0.0 (all interfaces)
**Max connections:** 5 (development), 20 (production)
**JWT expiration:** 7 days
**CORS:** Configured for frontend origin

### Database Configuration

**podman-compose.yml:**
```yaml
services:
  postgres:
    image: docker.io/library/postgres:18
    container_name: twitter_postgres
    environment:
      POSTGRES_USER: twitter_user
      POSTGRES_PASSWORD: twitter_pass
      POSTGRES_DB: twitter_db
    ports:
      - "5432:5432"
```

## Troubleshooting Guide

### Backend Issues

**Database Connection Errors**

Problem: Cannot connect to PostgreSQL

Solution:
```bash
podman ps
podman-compose logs postgres
podman-compose restart postgres
```

Check connection string in .env file.

**Migration Errors**

Problem: Migration fails on startup

Solution:
```bash
dropdb twitter_db && createdb twitter_db
cargo run
```

Or manually run migrations:
```bash
sqlx migrate run
```

**Port Already in Use**

Problem: Port 8000 is taken

Solution: Change port in src/main.rs:
```rust
let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
```

**Compilation Errors**

Problem: Cargo build fails

Solution:
```bash
cargo clean
cargo update
cargo build
```

### Frontend Issues

**API Connection Errors**

Problem: Cannot connect to backend

Solution:
- Verify backend is running on port 8000
- Check API_BASE_URL in src/lib/api.ts
- Check browser console for CORS errors
- Verify token in localStorage

**Authentication Issues**

Problem: Always redirected to login

Solution:
- Check token in localStorage
- Verify JWT_SECRET matches backend
- Check token expiration (7 days)
- Clear localStorage and login again

**Build Errors**

Problem: Frontend build fails

Solution:
```bash
rm -rf node_modules
npm install
npm run build
```

### Database Issues

**Slow Queries**

Problem: API responses are slow

Solution:
- Check indexes with `\d+ table_name` in psql
- Use EXPLAIN ANALYZE on slow queries
- Increase connection pool size
- Add missing indexes

**Data Inconsistency**

Problem: Counts don't match actual data

Solution:
- Run database integrity checks
- Verify CASCADE constraints
- Check for orphaned records
- Recalculate counts with direct queries

### Testing Issues

**Integration Tests Fail**

Problem: Tests timeout or fail

Solution:
- Ensure test database is running on port 5433
- Check DATABASE_URL environment variable
- Run tests with `--nocapture` for details
- Ensure clean database state

**E2E Tests Fail**

Problem: Playwright tests timeout

Solution:
- Verify backend on port 8000
- Verify frontend on port 5173
- Check browser console in headed mode
- Review screenshots in test-results/

**Performance Tests Fail**

Problem: k6 thresholds not met

Solution:
- Check backend logs for errors
- Monitor database connections
- Reduce virtual users
- Adjust thresholds
- Check system resources

## Known Limitations

### Current Limitations

1. **No Image/Media Upload**: Only text content supported
2. **No Direct Messages**: Private messaging not implemented
3. **No Notifications**: No real-time notification system
4. **No Email Verification**: Email not verified during registration
5. **No Password Reset**: Password recovery not implemented
6. **No Mentions/Hashtags**: @mentions and #hashtags not parsed
7. **No Search**: No search functionality for users or tweets
8. **No Trending Topics**: No trending hashtag tracking
9. **No User Blocking**: Cannot block users
10. **No Tweet Threads**: Reply chains not implemented
11. **No Pinned Tweets**: Cannot pin tweets to profile
12. **No Profile Pictures**: Avatar images not supported
13. **No Tweet Editing**: Cannot edit tweets after posting
14. **No Tweet Scheduling**: Cannot schedule tweets
15. **No Analytics**: No metrics dashboard

### Technical Constraints

**Character Limits:**
- Tweet content: 280 characters
- Comment content: 280 characters
- Username: 3-50 characters
- Display name: 100 characters
- Bio: 500 characters

**Rate Limits:**
- No rate limiting currently implemented
- Recommended: 300 requests per 15 minutes per user

**Pagination:**
- Feed default limit: 20 tweets
- Max limit: 100 tweets per request
- Offset-based pagination (not cursor-based)

**Security:**
- JWT tokens in localStorage (vulnerable to XSS)
- No CSRF protection
- No rate limiting
- No IP-based throttling

### Browser Support

**Supported Browsers:**
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

**Mobile Support:**
- iOS Safari 14+
- Chrome Mobile 90+
- Android WebView 90+

### Database Limits

**PostgreSQL:**
- Max connections: 100 (configurable)
- Connection pool: 5-20 connections
- Query timeout: 30 seconds
- Max table size: Unlimited (monitor performance at 10M+ rows)

## Future Enhancements

### Planned Features

1. **Image Upload**: AWS S3 integration for media
2. **Direct Messaging**: Private messaging between users
3. **Notifications**: Real-time notifications with WebSockets
4. **Email Verification**: Verify email during registration
5. **Password Reset**: Email-based password recovery
6. **Search**: Full-text search for users and tweets
7. **Mentions/Hashtags**: Parse and link @mentions and #hashtags
8. **User Blocking**: Block and mute users
9. **Tweet Threads**: Reply to tweets with threading
10. **Profile Pictures**: Upload and display avatars

### Performance Improvements

1. **Redis Caching**: Cache feeds and user profiles
2. **Database Read Replicas**: Scale read operations
3. **CDN Integration**: Static asset delivery
4. **GraphQL API**: Reduce over-fetching
5. **Cursor-based Pagination**: Better infinite scroll
6. **Connection Pooling**: Optimize database connections

### Security Enhancements

1. **Rate Limiting**: Implement per-user rate limits
2. **CSRF Protection**: Add CSRF tokens
3. **Content Moderation**: Flag inappropriate content
4. **Two-Factor Authentication**: Add 2FA support
5. **OAuth Integration**: Social login (Google, GitHub)

---

## Quick Reference

### API Base URL
```
http://localhost:8000/api
```

### Authentication Header
```
Authorization: Bearer <token>
```

### Key Endpoints
```
POST   /api/auth/register
POST   /api/auth/login
POST   /api/auth/logout
GET    /api/users/:id
PUT    /api/users/:id
POST   /api/users/:id/follow
DELETE /api/users/:id/follow
GET    /api/users/:id/followers
GET    /api/users/:id/following
POST   /api/tweets
GET    /api/tweets/:id
DELETE /api/tweets/:id
GET    /api/tweets/feed
GET    /api/tweets/user/:userId
POST   /api/tweets/:id/like
DELETE /api/tweets/:id/like
POST   /api/tweets/:id/retweet
DELETE /api/tweets/:id/retweet
POST   /api/tweets/:id/comments
GET    /api/tweets/:id/comments
DELETE /api/comments/:id
```

### Error Codes
```
400 - Bad Request (validation error)
401 - Unauthorized (missing/invalid token)
403 - Forbidden (insufficient permissions)
404 - Not Found (resource not found)
500 - Internal Server Error
```

### Frontend Routes
```
/login            - Authentication page
/                 - Home feed
/profile/:userId  - User profile
/tweet/:tweetId   - Tweet detail
```

### Scripts
```bash
./start.sh                   # Start backend and database
./stop.sh                    # Stop services
./test.sh                    # Test API manually
./run-integration-tests.sh   # Run integration tests
./run-e2e-tests.sh           # Run E2E tests
./run-k6-tests.sh            # Run performance tests
cargo test                   # Run unit tests
npm run dev                  # Start frontend dev server
npm run build                # Build frontend for production
```

---

**Documentation Version:** 1.0
**Last Updated:** 2026-02-09
**Application Version:** 1.0.0
