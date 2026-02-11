# Quick Start Guide

This guide will help you get the Twitter Clone backend up and running quickly.

## Prerequisites

1. Rust 1.93 or later
2. PostgreSQL 14 or later
3. Podman and podman-compose (or Docker)

## Installation Steps

### 1. Clone and Navigate

```bash
cd /private/tmp/test
```

### 2. Start PostgreSQL

```bash
chmod +x start.sh stop.sh test.sh
./start.sh
```

Or manually:
```bash
podman-compose up -d
```

### 3. Configure Environment

```bash
cp .env.template .env
```

Edit `.env` if needed to match your PostgreSQL configuration.

### 4. Build and Run

```bash
cargo build --release
cargo run --release
```

The server will start on `http://localhost:8000`.

### 5. Test the API

In a new terminal:
```bash
./test.sh
```

Or test manually with curl:

```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","email":"alice@test.com","password":"password123"}'
```

## Using Make

```bash
make db-up
make run
make test
```

## Stop the Server

Press `Ctrl+C` to stop the Rust server.

Stop PostgreSQL:
```bash
./stop.sh
```

Or:
```bash
make db-down
```

## API Testing Flow

1. Register a user
2. Login to get JWT token
3. Create tweets with the token
4. Follow other users
5. Get your feed
6. Like and retweet tweets
7. Add comments

## Troubleshooting

### Database Connection Issues

Make sure PostgreSQL is running:
```bash
podman ps
```

Check logs:
```bash
podman-compose logs postgres
```

### Port Already in Use

Change the port in `src/main.rs` if 8000 is taken.

### Migration Errors

Reset the database:
```bash
dropdb twitter
createdb twitter
cargo run
```

## Next Steps

- Check out the full API documentation in README.md
- Explore the code structure
- Run the test suite with `cargo test`
- Format code with `cargo fmt`
- Lint with `cargo clippy`
