# Twitter-like App Spec

## Functional Requirements
- Post tweets (username + content, 1-280 chars)
- View timeline (newest first)
- Delete own tweets
- Simple username via localStorage (no auth)

## Technical Stack
- Backend: Rust edition 2024, Actix-web, in-memory storage
- Frontend: React 19, TanStack Router, TanStack Query, Vite
- Scripts: run.sh / stop.sh with PID files

## API
- GET /api/tweets -> 200 Vec<Tweet>
- POST /api/tweets { username, content } -> 201 Tweet
- DELETE /api/tweets/{id}?username=X -> 204/403/404
- GET /api/health -> 200

## Tweet Schema
{ id: uuid, username: string, content: string, created_at: rfc3339 }

## Frontend
- Single route (/) with TweetForm + TweetList
- TanStack Query for server state
- Plain CSS styling
