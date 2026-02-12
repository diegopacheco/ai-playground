# Code Review - 2026-02-12

## Backend

### Architecture
The backend uses Axum with SQLx and SQLite. The separation into `auth`, `models`, `errors`, `routes`, and `db` modules is clean and well-structured. The `lib.rs` exposes `build_app` and `create_test_pool` for integration tests, which is a solid pattern.

### Findings

#### B-01: N+1 Query Pattern in `enrich_post` (Performance)
**File**: `backend/src/routes/posts.rs` lines 8-37
**Severity**: Medium
The `enrich_post` function executes 3 separate queries per post (username, likes count, liked_by_me). For `get_timeline` and `get_user_posts`, this is called in a loop, creating an N+1 query problem. For 50 posts, this is 150+ queries.

**Suggestion**: Use a single SQL query with JOINs and subqueries to fetch all post data at once.

#### B-02: Missing Pagination on `get_user_posts` (Feature Gap)
**File**: `backend/src/routes/posts.rs` line 199
**Severity**: Low
`get_user_posts` has no LIMIT clause, unlike `get_timeline` which limits to 50. A user with thousands of posts would return all of them.

#### B-03: `content.len()` Counts Bytes Not Characters (Logic)
**File**: `backend/src/routes/posts.rs` line 67
**Severity**: Low
`body.content.len()` measures byte length, not character count. Multi-byte UTF-8 characters (emojis, CJK) would hit the 280 limit earlier than expected. Use `.chars().count()` for Unicode-correct character counting.

#### B-04: No Rate Limiting on Auth Endpoints (Security/Performance)
**File**: `backend/src/routes/auth.rs`
**Severity**: Medium
No rate limiting on `/api/auth/register` or `/api/auth/login`. Allows brute-force attacks and mass account creation.

#### B-05: `get_followers` and `get_following` Missing Auth Extraction (Inconsistency)
**File**: `backend/src/routes/users.rs` lines 49-77
**Severity**: Low
These endpoints do not extract `Claims` from the request, unlike all other protected endpoints. They are behind the auth middleware so authentication is enforced, but the handlers ignore who is making the request. This is functionally correct but inconsistent.

#### B-06: CORS Difference Between `main.rs` and `lib.rs` (Inconsistency)
**File**: `backend/src/main.rs` line 21 vs `backend/src/lib.rs` line 76
**Severity**: Info
`main.rs` restricts CORS to `http://localhost:5173`, while `lib.rs` (test config) uses `Any` for all origins. This is acceptable for testing but worth noting.

#### B-07: `delete_post` Does Not Use a Transaction (Data Integrity)
**File**: `backend/src/routes/posts.rs` lines 111-138
**Severity**: Low
Deleting likes and then the post are two separate queries without a transaction. If the server crashes between them, orphaned state could remain.

### Positive Observations
- Error types are well-defined with proper HTTP status code mapping
- Password hashing uses Argon2 with proper salt generation
- JWT auth middleware is clean and correctly strips the Bearer prefix
- `From<User> for UserResponse` properly excludes the password hash
- Input validation for empty fields, email format, and post content length
- Self-follow prevention in `follow_user`
- Duplicate like/follow detection returns 409 Conflict

---

## Frontend

### Architecture
React 19 with TypeScript, TanStack Router, TanStack Query, and Tailwind CSS. Clean separation between API layer, auth utilities, hooks, pages, and components.

### Findings

#### F-01: `getUser()` Parses JSON Without Error Handling (Robustness)
**File**: `frontend/src/auth.ts` line 13
**Severity**: Low
`JSON.parse(raw)` could throw if localStorage contains corrupted data. A try-catch would prevent the entire app from crashing.

#### F-02: No Token Expiration Check on Frontend (Auth Gap)
**File**: `frontend/src/auth.ts` line 27
**Severity**: Medium
`isAuthenticated()` only checks if a token exists in localStorage. It does not check whether the JWT has expired. Users with expired tokens will be redirected to the home page but all API calls will fail with 401. The app should decode the token and check the `exp` claim, or handle 401 responses globally by clearing auth and redirecting to login.

#### F-03: API Base URL Assumes Vite Proxy (Configuration)
**File**: `frontend/src/api.ts` line 9
**Severity**: Info
`const BASE = "/api"` relies on a Vite dev server proxy to forward requests to the backend. This is correct for development but means a production deployment needs a reverse proxy or the BASE URL needs to change.

#### F-04: Optimistic Update Only Covers Timeline Query (Partial Coverage)
**File**: `frontend/src/hooks/usePosts.ts` lines 41-56
**Severity**: Low
The `useToggleLike` hook only applies optimistic updates to the `["timeline"]` query data. If the user likes a post from the profile page or post detail page, the optimistic update will not reflect there. The `onSettled` invalidation covers eventual consistency, but there is a brief UI delay on non-timeline views.

#### F-05: `useParams` Cast With `as` (Type Safety)
**File**: `frontend/src/pages/ProfilePage.tsx` line 9, `PostDetailPage.tsx` line 6
**Severity**: Info
Using `as { userId: string }` bypasses TanStack Router's type safety. This works but is fragile if routes change.

### Positive Observations
- TanStack Query with proper query key design enables cache invalidation
- Optimistic updates for likes provide instant UI feedback
- Character counter with color coding in PostComposer
- Clean loading and empty states
- Proper form validation with disabled submit buttons during pending state
- Auth guard via TanStack Router `beforeLoad` redirect

---

## Database

### Findings

#### D-01: Schema Has `CHECK(length(content) <= 280)` But Backend Also Validates (Redundancy)
**File**: `db/schema.sql` line 14
**Severity**: Info
Both the SQL schema and the Rust backend validate content length. Defense in depth is good, but note the SQL `length()` counts characters while Rust `len()` counts bytes (see B-03).

#### D-02: No Index on `follows.follower_id` (Performance)
**File**: `db/schema.sql`
**Severity**: Low
There is an index on `follows.following_id` but not on `follows.follower_id`. The timeline query uses `SELECT following_id FROM follows WHERE follower_id = ?`, which would benefit from an index on `follower_id`. The composite primary key (follower_id, following_id) likely serves as an implicit index on `follower_id` in SQLite, so this is minor.

#### D-03: No `PRAGMA foreign_keys = ON` in Backend Code (Data Integrity)
**File**: `backend/src/db.rs`
**Severity**: Medium
The `db/schema.sql` file includes `PRAGMA foreign_keys = ON` but the backend `db.rs` and `lib.rs` do not enable this pragma. SQLite has foreign keys disabled by default, so the foreign key constraints defined in the CREATE TABLE statements are not enforced at runtime. The `delete_post` handler manually deletes likes before the post, which works around this, but other foreign key violations would go undetected.

### Positive Observations
- Composite primary keys on `likes` and `follows` prevent duplicates at the DB level
- Schema uses `CHECK(follower_id != following_id)` to prevent self-follows at DB level
- Proper indexes on frequently queried columns
