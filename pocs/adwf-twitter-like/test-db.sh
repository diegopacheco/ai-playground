#!/bin/bash

echo "Testing PostgreSQL Database Setup"
echo "=================================="
echo ""

echo "1. Starting PostgreSQL container..."
./start-db.sh
if [ $? -eq 0 ]; then
  echo "   PostgreSQL started successfully!"
else
  echo "   Failed to start PostgreSQL"
  exit 1
fi
echo ""

echo "2. Creating schema..."
./create-schema.sh
if [ $? -eq 0 ]; then
  echo "   Schema created successfully!"
else
  echo "   Failed to create schema"
  exit 1
fi
echo ""

echo "3. Verifying tables..."
podman exec twitter_postgres psql -U twitter_user -d twitter_db -c "\dt"
echo ""

echo "4. Verifying indexes..."
podman exec twitter_postgres psql -U twitter_user -d twitter_db -c "\di"
echo ""

echo "5. Inserting test data..."
podman exec twitter_postgres psql -U twitter_user -d twitter_db -c "
INSERT INTO users (username, email, password_hash, display_name, bio)
VALUES ('testuser', 'test@twitter.com', 'hashed_password', 'Test User', 'This is a test user');
"
echo ""

echo "6. Querying test data..."
podman exec twitter_postgres psql -U twitter_user -d twitter_db -c "SELECT id, username, email, display_name FROM users;"
echo ""

echo "7. Testing tweet creation..."
podman exec twitter_postgres psql -U twitter_user -d twitter_db -c "
INSERT INTO tweets (user_id, content)
VALUES (1, 'This is my first tweet!');
"
echo ""

echo "8. Verifying tweet..."
podman exec twitter_postgres psql -U twitter_user -d twitter_db -c "
SELECT t.id, t.content, t.created_at, u.username
FROM tweets t
JOIN users u ON t.user_id = u.id;
"
echo ""

echo "Database setup test completed successfully!"
