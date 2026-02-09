# Blog Platform - Design Document

## System Architecture

The Blog Platform is a full-stack web application with three layers:

```
+-------------------+       +-------------------+       +-------------------+
|    Frontend       | ----> |    Backend         | ----> |    Database        |
|    React 19       |       |    Rust / Axum     |       |    PostgreSQL 17   |
|    Port 3000      |       |    Port 8080       |       |    Port 5432       |
+-------------------+       +-------------------+       +-------------------+
```

The frontend runs on port 3000 via Vite dev server and proxies all `/api` requests to the backend at port 8080. The backend connects directly to PostgreSQL using sqlx with a connection pool of up to 10 connections. CORS is enabled on the backend with permissive settings (allow any origin, method, and header). Tables are auto-created on backend startup using `CREATE TABLE IF NOT EXISTS` statements.

## Data Model

All entities use UUID v4 as primary keys. Timestamps use `TIMESTAMP` (NaiveDateTime in Rust, string in TypeScript).

### Users

| Column     | Type      | Constraints       |
|------------|-----------|-------------------|
| id         | UUID      | PRIMARY KEY       |
| name       | TEXT      | NOT NULL          |
| email      | TEXT      | NOT NULL, UNIQUE  |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() |

### Posts

| Column     | Type      | Constraints       |
|------------|-----------|-------------------|
| id         | UUID      | PRIMARY KEY       |
| title      | TEXT      | NOT NULL          |
| content    | TEXT      | NOT NULL          |
| author     | TEXT      | NOT NULL          |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() |
| updated_at | TIMESTAMP | NOT NULL, DEFAULT NOW() |

### Comments

| Column     | Type      | Constraints                          |
|------------|-----------|--------------------------------------|
| id         | UUID      | PRIMARY KEY                          |
| content    | TEXT      | NOT NULL                             |
| author     | TEXT      | NOT NULL                             |
| post_id    | UUID      | NOT NULL, FK -> posts(id) ON DELETE CASCADE |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW()              |

Deleting a post cascades to delete all its comments. The `author` field on posts and comments is a plain text string, not a foreign key to the users table.

## API Endpoints

All endpoints are prefixed with `/api` and served on port 8080. Request and response bodies use JSON. Rust models serialize with camelCase field names via `serde(rename_all = "camelCase")`.

### Users

| Method | Path              | Description        | Request Body              | Response     |
|--------|-------------------|--------------------|---------------------------|--------------|
| POST   | `/api/users`      | Create a user      | `{ name, email }`         | 201: User    |
| GET    | `/api/users`      | List all users     | -                         | 200: User[]  |
| GET    | `/api/users/{id}` | Get a user by ID   | -                         | 200: User    |

### Posts

| Method | Path              | Description        | Request Body                   | Response     |
|--------|-------------------|--------------------|--------------------------------|--------------|
| POST   | `/api/posts`      | Create a post      | `{ title, content, author }`   | 201: Post    |
| GET    | `/api/posts`      | List all posts     | -                              | 200: Post[]  |
| GET    | `/api/posts/{id}` | Get a post by ID   | -                              | 200: Post    |
| PUT    | `/api/posts/{id}` | Update a post      | `{ title?, content?, author? }`| 200: Post    |
| DELETE | `/api/posts/{id}` | Delete a post      | -                              | 200: { deleted: true } |

### Comments

| Method | Path                              | Description               | Request Body             | Response        |
|--------|-----------------------------------|---------------------------|--------------------------|-----------------|
| POST   | `/api/posts/{post_id}/comments`   | Create a comment on post  | `{ content, author }`    | 201: Comment    |
| GET    | `/api/posts/{post_id}/comments`   | List comments for a post  | -                        | 200: Comment[]  |
| DELETE | `/api/comments/{id}`              | Delete a comment by ID    | -                        | 200: { deleted: true } |

Error responses return `{ "error": "<message>" }` with the appropriate HTTP status code (404, 500).

## Frontend Routes and Pages

The frontend uses TanStack Router for client-side routing. A shared root layout provides a navigation bar with links to Home, New Post, and Profile.

| Route                   | Page             | Description                                          |
|-------------------------|------------------|------------------------------------------------------|
| `/`                     | HomePage         | Lists all posts ordered by newest first              |
| `/posts/create`         | CreatePostPage   | Form to create a new post (title, author, content)   |
| `/posts/$postId`        | PostDetailPage   | Shows full post, comments list, and add comment form |
| `/posts/$postId/edit`   | EditPostPage     | Form to edit post title and content                  |
| `/profile`              | ProfilePage      | Lists all registered users                           |

The `useApi.ts` hook module wraps all API calls using TanStack Query, providing:

- `usePosts()` - fetch all posts
- `usePost(id)` - fetch a single post
- `useCreatePost()` - mutation to create a post
- `useUpdatePost(id)` - mutation to update a post
- `useDeletePost()` - mutation to delete a post
- `useComments(postId)` - fetch comments for a post
- `useCreateComment(postId)` - mutation to add a comment
- `useUsers()` - fetch all users

All mutations automatically invalidate the relevant query cache on success.

## Technology Stack

### Backend

| Technology         | Version | Purpose                        |
|--------------------|---------|--------------------------------|
| Rust               | 2024 edition | Language                  |
| Axum               | 0.8     | HTTP framework                 |
| Tokio              | 1.x     | Async runtime                  |
| sqlx               | 0.8     | PostgreSQL driver (async)      |
| uuid               | 1.x     | UUID v4 generation             |
| chrono             | 0.4     | Timestamp handling             |
| serde / serde_json | 1.x     | JSON serialization             |
| tower-http         | 0.6     | CORS middleware                |
| tracing            | 0.1     | Structured logging             |

### Frontend

| Technology           | Version | Purpose                      |
|----------------------|---------|------------------------------|
| React                | 19.x    | UI framework                 |
| TanStack Router      | 1.92.x  | Client-side routing          |
| TanStack Query       | 5.62.x  | Server state management      |
| Tailwind CSS         | 4.x     | Utility-first CSS            |
| Vite                 | 6.x     | Build tool and dev server    |
| TypeScript           | 5.7     | Type safety                  |

### Database

| Technology   | Version | Purpose                |
|--------------|---------|------------------------|
| PostgreSQL   | 17      | Relational database    |

### Infrastructure

| Tool             | Purpose                         |
|------------------|---------------------------------|
| podman-compose   | Container orchestration for DB  |
| Bun              | JavaScript runtime and package manager |

## How to Run

### 1. Start the Database

From the `backend/` directory:

```bash
./start.sh
```

This runs `podman-compose up -d` to start PostgreSQL 17 in a container named `blog-postgres`, waits for it to be ready, then starts the backend. The database credentials are `postgres:postgres` and the database name is `blog_platform`.

Alternatively, from the `db/` directory:

```bash
./start-db.sh
```

This starts PostgreSQL as a standalone podman container named `blogdb-postgres` with credentials `bloguser:blogpass` and database `blogdb`.

### 2. Start the Backend

From the `backend/` directory:

```bash
cargo run
```

The server starts on `http://0.0.0.0:8080`. It auto-creates the `users`, `posts`, and `comments` tables on startup. The default database URL is `postgres://postgres:postgres@localhost:5432/blog_platform` and can be overridden with the `DATABASE_URL` environment variable.

### 3. Start the Frontend

From the `frontend/` directory:

```bash
bun install
bun dev
```

The Vite dev server starts on `http://localhost:3000` and proxies `/api` requests to `http://localhost:8080`.

### 4. Stop the Database

From the `backend/` directory:

```bash
./stop.sh
```

This runs `podman-compose down` to stop and remove the PostgreSQL container.

### 5. Run API Tests

From the `backend/` directory:

```bash
./test.sh
```

This exercises the full CRUD flow via curl: creates a user, creates a post, updates the post, creates a comment, and lists comments.
