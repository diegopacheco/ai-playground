# Development Guide

## Project Structure

The project follows a modular architecture:

```
src/
├── main.rs           - Entry point, server setup
├── lib.rs            - Library exports for testing
├── config.rs         - Configuration from environment
├── state.rs          - Application state (DB pool, config)
├── error.rs          - Error types and handling
├── middleware/       - Authentication and request processing
├── models/           - Data models and validation
├── handlers/         - Business logic for each endpoint
└── routes/           - Route definitions and middleware setup
```

## Code Organization

### Models Layer
- Define data structures
- Implement validation rules
- Serialize/deserialize for JSON
- Map to database rows

### Handlers Layer
- Extract and validate request data
- Call database operations
- Transform data for responses
- Handle business logic errors

### Routes Layer
- Define HTTP routes
- Apply middleware
- Connect handlers to endpoints

### Middleware Layer
- JWT token verification
- Request/response processing
- Authentication enforcement

## Database Operations

All database operations use SQLx with async/await:

```rust
let user = sqlx::query_as::<_, User>(
    "SELECT * FROM users WHERE id = $1"
)
.bind(user_id)
.fetch_one(&state.db)
.await?;
```

### Best Practices

1. Use parameterized queries to prevent SQL injection
2. Handle errors with proper error types
3. Use transactions for multi-step operations
4. Index frequently queried columns
5. Use connection pooling

## Authentication Flow

1. User registers or logs in
2. Server generates JWT token with user ID
3. Client stores token
4. Client sends token in Authorization header
5. Middleware extracts and validates token
6. Handler accesses user ID from request extensions

## Error Handling

The application uses a custom `AppError` type that implements `IntoResponse`:

```rust
pub enum AppError {
    Database(sqlx::Error),
    Authentication(String),
    Authorization(String),
    NotFound(String),
    BadRequest(String),
    Internal(anyhow::Error),
}
```

Handlers return `Result<T, AppError>` which automatically converts to HTTP responses.

## Testing Strategy

### Unit Tests
Test individual functions and validation:
```rust
cargo test test_register_request_validation
```

### Integration Tests
Test full API endpoints with a test database:
```rust
cargo test test_user_registration_and_login
```

### Manual Testing
Use the `test.sh` script to test the live API.

## Adding New Features

### 1. Add Database Migration

Create a new migration file in `migrations/`:
```sql
-- migrations/YYYYMMDDHHMMSS_add_feature.sql
CREATE TABLE new_feature (...);
```

### 2. Define Models

Add to `src/models/`:
```rust
pub struct NewFeature {
    pub id: i32,
    // fields
}
```

### 3. Create Handlers

Add to `src/handlers/`:
```rust
pub async fn handle_feature(
    State(state): State<Arc<AppState>>,
    // parameters
) -> Result<Json<Response>> {
    // implementation
}
```

### 4. Add Routes

Update `src/routes/mod.rs`:
```rust
.route("/feature", post(handlers::handle_feature))
```

### 5. Test

Write tests and run:
```bash
cargo test
cargo clippy
cargo fmt
```

## Performance Considerations

1. Use connection pooling (configured in main.rs)
2. Add database indexes for queries
3. Use prepared statements via SQLx
4. Implement pagination for list endpoints
5. Cache frequently accessed data

## Security

1. Passwords hashed with bcrypt
2. JWT tokens for stateless auth
3. SQL injection prevention via parameterized queries
4. CORS configured for frontend
5. Input validation on all endpoints

## Logging

Configure logging via environment:
```bash
RUST_LOG=twitter_clone=debug,tower_http=debug cargo run
```

Levels: error, warn, info, debug, trace

## Database Migrations

Run migrations:
```bash
sqlx migrate run
```

Revert last migration:
```bash
sqlx migrate revert
```

Add new migration:
```bash
sqlx migrate add migration_name
```

## Environment Variables

Required:
- `DATABASE_URL` - PostgreSQL connection string
- `JWT_SECRET` - Secret for JWT signing

Optional:
- `RUST_LOG` - Logging configuration

## Common Commands

```bash
cargo build              # Build debug version
cargo build --release    # Build optimized version
cargo run                # Run debug version
cargo run --release      # Run optimized version
cargo test               # Run all tests
cargo fmt                # Format code
cargo clippy             # Lint code
cargo clean              # Clean build artifacts
```

## Troubleshooting

### Compilation Errors

Update dependencies:
```bash
cargo update
```

Clean and rebuild:
```bash
cargo clean && cargo build
```

### Database Connection Errors

Check PostgreSQL is running:
```bash
podman ps
```

Verify connection string in `.env`

### Migration Errors

Reset database:
```bash
dropdb twitter && createdb twitter
```

### Port Conflicts

Change port in `src/main.rs`:
```rust
let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
```

## Contributing

1. Format code: `cargo fmt`
2. Run linter: `cargo clippy`
3. Run tests: `cargo test`
4. Update documentation
5. Follow Rust API guidelines
