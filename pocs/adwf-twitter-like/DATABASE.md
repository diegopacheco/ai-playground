# Twitter Clone - Database Setup

## Overview
PostgreSQL 18 database schema for a Twitter clone application with users, tweets, follows, likes, retweets, and comments.

## Database Configuration
- **Database Name**: twitter_db
- **User**: twitter_user
- **Password**: twitter_pass
- **Port**: 5432 (localhost)
- **Container**: twitter_postgres

## Database Schema

### Tables

**users**
- Stores user account information
- Unique constraints on username and email
- Indexed on username and email for fast lookups

**tweets**
- Stores user tweets (max 280 characters)
- Foreign key to users table with CASCADE delete
- Indexed on user_id and created_at for feed queries

**follows**
- Junction table for user-to-user follow relationships
- Composite primary key (follower_id, following_id)
- Check constraint prevents self-following
- Indexed on both follower_id and following_id

**likes**
- Junction table for user-tweet likes
- Composite primary key (user_id, tweet_id)
- Indexed on both user_id and tweet_id

**retweets**
- Junction table for user-tweet retweets
- Composite primary key (user_id, tweet_id)
- Indexed on both user_id and tweet_id

**comments**
- Stores comments on tweets (max 280 characters)
- Foreign keys to both users and tweets with CASCADE delete
- Indexed on tweet_id, user_id, and created_at

## Scripts

### start-db.sh
Starts a PostgreSQL 18 container using podman with automatic health checking.
```bash
./start-db.sh
```

### stop-db.sh
Stops and removes the PostgreSQL container.
```bash
./stop-db.sh
```

### create-schema.sh
Executes the schema.sql file to create all tables and indexes.
```bash
./create-schema.sh
```

### run-sql-client.sh
Opens an interactive psql session to the database.
```bash
./run-sql-client.sh
```

### test-db.sh
Comprehensive test script that starts the database, creates the schema, and runs verification tests.
```bash
./test-db.sh
```

## Usage

1. Make scripts executable:
```bash
chmod +x *.sh
```

2. Start the database:
```bash
./start-db.sh
```

3. Create the schema:
```bash
./create-schema.sh
```

4. Connect to the database:
```bash
./run-sql-client.sh
```

5. Stop the database when done:
```bash
./stop-db.sh
```

## Query Patterns

### Get User Feed
```sql
SELECT t.*, u.username, u.display_name
FROM tweets t
JOIN users u ON t.user_id = u.id
WHERE t.user_id IN (
    SELECT following_id FROM follows WHERE follower_id = $1
)
ORDER BY t.created_at DESC
LIMIT 50;
```

### Get User Profile with Tweet Count
```sql
SELECT u.*, COUNT(t.id) as tweet_count
FROM users u
LEFT JOIN tweets t ON u.id = t.user_id
WHERE u.id = $1
GROUP BY u.id;
```

### Get Tweet with Like and Retweet Counts
```sql
SELECT
    t.*,
    u.username,
    u.display_name,
    COUNT(DISTINCT l.user_id) as like_count,
    COUNT(DISTINCT r.user_id) as retweet_count,
    COUNT(DISTINCT c.id) as comment_count
FROM tweets t
JOIN users u ON t.user_id = u.id
LEFT JOIN likes l ON t.id = l.tweet_id
LEFT JOIN retweets r ON t.id = r.tweet_id
LEFT JOIN comments c ON t.id = c.tweet_id
WHERE t.id = $1
GROUP BY t.id, u.username, u.display_name;
```

### Get User Followers
```sql
SELECT u.id, u.username, u.display_name, u.bio
FROM users u
JOIN follows f ON u.id = f.follower_id
WHERE f.following_id = $1;
```

### Get User Following
```sql
SELECT u.id, u.username, u.display_name, u.bio
FROM users u
JOIN follows f ON u.id = f.following_id
WHERE f.follower_id = $1;
```

## Performance Considerations

### Indexes
- User lookups by username/email are O(log n) via B-tree indexes
- Tweet feed queries benefit from user_id and created_at indexes
- Follow relationships use bidirectional indexes for fast lookups
- Like/retweet counts use indexes on tweet_id
- Comment queries use tweet_id and created_at indexes

### Constraints
- Foreign keys ensure referential integrity
- CASCADE deletes maintain data consistency
- Unique constraints prevent duplicate follows/likes/retweets
- Check constraint prevents self-following

### Recommendations
- Use connection pooling (10-20 connections recommended)
- Enable prepared statements for repeated queries
- Monitor slow query log for queries > 100ms
- Consider partitioning tweets table if data exceeds 10M rows
- Add covering indexes if specific query patterns emerge
