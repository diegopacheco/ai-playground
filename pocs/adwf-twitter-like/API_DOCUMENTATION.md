# Twitter Clone API Documentation

Base URL: `http://localhost:8000`

All authenticated endpoints require a JWT token in the Authorization header:
```
Authorization: Bearer <token>
```

## Authentication Endpoints

### Register User
```
POST /api/auth/register
```

Request Body:
```json
{
  "username": "string (3-50 chars)",
  "email": "string (valid email)",
  "password": "string (min 6 chars)"
}
```

Response: 201 Created
```json
{
  "token": "jwt_token_string",
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

### Login
```
POST /api/auth/login
```

Request Body:
```json
{
  "username": "string",
  "password": "string"
}
```

Response: 200 OK
```json
{
  "token": "jwt_token_string",
  "user": { /* user object */ }
}
```

### Logout
```
POST /api/auth/logout
```

Response: 204 No Content

## User Endpoints

All user endpoints require authentication.

### Get User Profile
```
GET /api/users/:id
```

Response: 200 OK
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

### Update User Profile
```
PUT /api/users/:id
```

Request Body:
```json
{
  "display_name": "string (max 100 chars, optional)",
  "bio": "string (max 500 chars, optional)"
}
```

Response: 200 OK
```json
{
  /* updated user object */
}
```

### Get Followers
```
GET /api/users/:id/followers
```

Response: 200 OK
```json
[
  {
    "id": 2,
    "username": "bob",
    "display_name": "Bob Jones",
    /* other user fields */
  }
]
```

### Get Following
```
GET /api/users/:id/following
```

Response: 200 OK
```json
[
  {
    "id": 3,
    "username": "charlie",
    /* other user fields */
  }
]
```

### Follow User
```
POST /api/users/:id/follow
```

Response: 201 Created

### Unfollow User
```
DELETE /api/users/:id/follow
```

Response: 204 No Content

## Tweet Endpoints

All tweet endpoints require authentication.

### Create Tweet
```
POST /api/tweets
```

Request Body:
```json
{
  "content": "string (1-280 chars)"
}
```

Response: 201 Created
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

### Get Tweet
```
GET /api/tweets/:id
```

Response: 200 OK
```json
{
  /* tweet response object */
}
```

### Delete Tweet
```
DELETE /api/tweets/:id
```

Response: 204 No Content

### Get Feed
```
GET /api/tweets/feed?limit=20&offset=0
```

Query Parameters:
- `limit`: number of tweets (default: 20)
- `offset`: pagination offset (default: 0)

Response: 200 OK
```json
[
  {
    /* tweet response objects */
  }
]
```

### Get User Tweets
```
GET /api/tweets/user/:userId?limit=20&offset=0
```

Query Parameters:
- `limit`: number of tweets (default: 20)
- `offset`: pagination offset (default: 0)

Response: 200 OK
```json
[
  {
    /* tweet response objects */
  }
]
```

### Like Tweet
```
POST /api/tweets/:id/like
```

Response: 201 Created

### Unlike Tweet
```
DELETE /api/tweets/:id/like
```

Response: 204 No Content

### Retweet
```
POST /api/tweets/:id/retweet
```

Response: 201 Created

### Remove Retweet
```
DELETE /api/tweets/:id/retweet
```

Response: 204 No Content

## Comment Endpoints

All comment endpoints require authentication.

### Create Comment
```
POST /api/tweets/:id/comments
```

Request Body:
```json
{
  "content": "string (1-280 chars)"
}
```

Response: 201 Created
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
```
GET /api/tweets/:id/comments
```

Response: 200 OK
```json
[
  {
    /* comment response objects */
  }
]
```

### Delete Comment
```
DELETE /api/comments/:id
```

Response: 204 No Content

## Error Responses

All endpoints may return error responses:

### 400 Bad Request
```json
{
  "error": "Invalid input or validation error"
}
```

### 401 Unauthorized
```json
{
  "error": "Missing or invalid authentication token"
}
```

### 403 Forbidden
```json
{
  "error": "Insufficient permissions"
}
```

### 404 Not Found
```json
{
  "error": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error"
}
```

## Testing with curl

### Register and Login
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","email":"alice@test.com","password":"password123"}'

TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"password123"}' \
  | jq -r '.token')
```

### Create Tweet
```bash
curl -X POST http://localhost:8000/api/tweets \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"content":"My first tweet!"}'
```

### Get Feed
```bash
curl -X GET http://localhost:8000/api/tweets/feed \
  -H "Authorization: Bearer $TOKEN"
```

### Follow User
```bash
curl -X POST http://localhost:8000/api/users/2/follow \
  -H "Authorization: Bearer $TOKEN"
```

### Like Tweet
```bash
curl -X POST http://localhost:8000/api/tweets/1/like \
  -H "Authorization: Bearer $TOKEN"
```

### Add Comment
```bash
curl -X POST http://localhost:8000/api/tweets/1/comments \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"content":"Great tweet!"}'
```
