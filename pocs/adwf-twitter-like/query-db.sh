#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage: ./query-db.sh 'SQL QUERY'"
  echo "Execute a SQL query against the twitter_db database"
  exit 1
fi

podman exec twitter_postgres psql -U twitter_user -d twitter_db -c "$1"
