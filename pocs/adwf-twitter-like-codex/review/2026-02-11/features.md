# Features 2026-02-11

- Create users via `POST /api/users`.
- Follow users via `POST /api/follows`.
- Create posts via `POST /api/posts` with 280 char cap.
- List posts via `GET /api/posts`.
- Like posts via `POST /api/posts/:id/likes`.
- Build timeline via `GET /api/timeline/:user_id`.
- Web client with user creation, post creation, list feed, and like action.
- DB schema for users, posts, likes, and follows.
- Unit tests, integration test script, UI test script, and K6 stress test script.
