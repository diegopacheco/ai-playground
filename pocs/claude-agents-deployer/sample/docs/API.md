# Blog Platform API Documentation

Base URL: `http://localhost:8080`

All request and response bodies use JSON. All IDs are UUIDs. All JSON responses use camelCase field names.

---

## Users API

### POST /api/users

Create a new user.

**Request Body**

| Field | Type   | Required |
|-------|--------|----------|
| name  | string | yes      |
| email | string | yes      |

**Response** `201 Created`

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "John Doe",
  "email": "john@test.com",
  "createdAt": "2025-01-15T10:30:00"
}
```

**curl**

```bash
curl -s -X POST http://localhost:8080/api/users \
  -H "Content-Type: application/json" \
  -d '{"name":"John Doe","email":"john@test.com"}'
```

---

### GET /api/users

List all users ordered by creation date (newest first).

**Response** `200 OK`

```json
[
  {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "name": "John Doe",
    "email": "john@test.com",
    "createdAt": "2025-01-15T10:30:00"
  }
]
```

**curl**

```bash
curl -s http://localhost:8080/api/users
```

---

### GET /api/users/:id

Get a single user by UUID.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| id        | UUID | User ID     |

**Response** `200 OK`

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "John Doe",
  "email": "john@test.com",
  "createdAt": "2025-01-15T10:30:00"
}
```

**curl**

```bash
curl -s http://localhost:8080/api/users/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

---

## Posts API

### POST /api/posts

Create a new blog post.

**Request Body**

| Field   | Type   | Required |
|---------|--------|----------|
| title   | string | yes      |
| content | string | yes      |
| author  | string | yes      |

**Response** `201 Created`

```json
{
  "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "title": "First Post",
  "content": "Hello World",
  "author": "John Doe",
  "createdAt": "2025-01-15T10:30:00",
  "updatedAt": "2025-01-15T10:30:00"
}
```

**curl**

```bash
curl -s -X POST http://localhost:8080/api/posts \
  -H "Content-Type: application/json" \
  -d '{"title":"First Post","content":"Hello World","author":"John Doe"}'
```

---

### GET /api/posts

List all posts ordered by creation date (newest first).

**Response** `200 OK`

```json
[
  {
    "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
    "title": "First Post",
    "content": "Hello World",
    "author": "John Doe",
    "createdAt": "2025-01-15T10:30:00",
    "updatedAt": "2025-01-15T10:30:00"
  }
]
```

**curl**

```bash
curl -s http://localhost:8080/api/posts
```

---

### GET /api/posts/:id

Get a single post by UUID.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| id        | UUID | Post ID     |

**Response** `200 OK`

```json
{
  "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "title": "First Post",
  "content": "Hello World",
  "author": "John Doe",
  "createdAt": "2025-01-15T10:30:00",
  "updatedAt": "2025-01-15T10:30:00"
}
```

**curl**

```bash
curl -s http://localhost:8080/api/posts/b2c3d4e5-f6a7-8901-bcde-f12345678901
```

---

### PUT /api/posts/:id

Update an existing post. All fields are optional; only provided fields are updated. The `updatedAt` timestamp is automatically refreshed.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| id        | UUID | Post ID     |

**Request Body**

| Field   | Type   | Required |
|---------|--------|----------|
| title   | string | no       |
| content | string | no       |
| author  | string | no       |

**Response** `200 OK`

```json
{
  "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "title": "Updated Title",
  "content": "Hello World",
  "author": "John Doe",
  "createdAt": "2025-01-15T10:30:00",
  "updatedAt": "2025-01-15T11:00:00"
}
```

**curl**

```bash
curl -s -X PUT http://localhost:8080/api/posts/b2c3d4e5-f6a7-8901-bcde-f12345678901 \
  -H "Content-Type: application/json" \
  -d '{"title":"Updated Title"}'
```

---

### DELETE /api/posts/:id

Delete a post by UUID. This also deletes all associated comments (cascade).

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| id        | UUID | Post ID     |

**Response** `200 OK`

```json
{
  "deleted": true
}
```

**curl**

```bash
curl -s -X DELETE http://localhost:8080/api/posts/b2c3d4e5-f6a7-8901-bcde-f12345678901
```

---

## Comments API

### POST /api/posts/:post_id/comments

Create a new comment on a post. The target post must exist.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| post_id   | UUID | Post ID     |

**Request Body**

| Field   | Type   | Required |
|---------|--------|----------|
| content | string | yes      |
| author  | string | yes      |

**Response** `201 Created`

```json
{
  "id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
  "content": "Great post!",
  "author": "Jane Doe",
  "postId": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "createdAt": "2025-01-15T11:00:00"
}
```

**curl**

```bash
curl -s -X POST http://localhost:8080/api/posts/b2c3d4e5-f6a7-8901-bcde-f12345678901/comments \
  -H "Content-Type: application/json" \
  -d '{"content":"Great post!","author":"Jane Doe"}'
```

---

### GET /api/posts/:post_id/comments

List all comments for a post ordered by creation date (newest first).

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| post_id   | UUID | Post ID     |

**Response** `200 OK`

```json
[
  {
    "id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
    "content": "Great post!",
    "author": "Jane Doe",
    "postId": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
    "createdAt": "2025-01-15T11:00:00"
  }
]
```

**curl**

```bash
curl -s http://localhost:8080/api/posts/b2c3d4e5-f6a7-8901-bcde-f12345678901/comments
```

---

### DELETE /api/comments/:id

Delete a comment by UUID.

**Path Parameters**

| Parameter | Type | Description  |
|-----------|------|--------------|
| id        | UUID | Comment ID   |

**Response** `200 OK`

```json
{
  "deleted": true
}
```

**curl**

```bash
curl -s -X DELETE http://localhost:8080/api/comments/c3d4e5f6-a7b8-9012-cdef-123456789012
```

---

## Error Responses

All endpoints return errors in a consistent JSON format.

### 404 Not Found

Returned when a resource does not exist.

```json
{
  "error": "User not found"
}
```

```json
{
  "error": "Post not found"
}
```

```json
{
  "error": "Comment not found"
}
```

### 500 Internal Server Error

Returned when a database or server error occurs.

```json
{
  "error": "error details from the server"
}
```

### 422 Unprocessable Entity

Returned when the request body is missing required fields or has invalid JSON.

---

## Field Reference

### User

| Field     | Type     | Description          |
|-----------|----------|----------------------|
| id        | UUID     | Unique identifier    |
| name      | string   | Full name            |
| email     | string   | Email address (unique)|
| createdAt | datetime | Creation timestamp   |

### Post

| Field     | Type     | Description            |
|-----------|----------|------------------------|
| id        | UUID     | Unique identifier      |
| title     | string   | Post title             |
| content   | string   | Post body              |
| author    | string   | Author name            |
| createdAt | datetime | Creation timestamp     |
| updatedAt | datetime | Last update timestamp  |

### Comment

| Field     | Type     | Description            |
|-----------|----------|------------------------|
| id        | UUID     | Unique identifier      |
| content   | string   | Comment body           |
| author    | string   | Author name            |
| postId    | UUID     | Parent post identifier |
| createdAt | datetime | Creation timestamp     |
