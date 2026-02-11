#!/bin/bash

echo "Setting up test database..."
./setup-test-db.sh

if [ $? -ne 0 ]; then
  echo "Failed to setup test database"
  exit 1
fi

export DATABASE_URL="postgres://postgres:postgres@localhost:5433/twitter_test"
export JWT_SECRET="test-secret-key-for-integration-tests"

echo "Running integration tests..."
cargo test --test integration_tests -- --nocapture --test-threads=1

test_result=$?

echo "Cleaning up test database..."
podman stop twitter_test_db
podman rm twitter_test_db

exit $test_result
