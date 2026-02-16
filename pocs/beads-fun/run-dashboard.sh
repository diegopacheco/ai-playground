#!/bin/bash
DB_PATH="backend/twitter.db"
if [ ! -f "$DB_PATH" ]; then
  echo "Database not found at $DB_PATH"
  echo "Run the backend first to create it."
  exit 1
fi
echo "=== Twitter App SQLite Dashboard ==="
echo ""
while true; do
  echo "Pick an option:"
  echo "  1) Show all tweets"
  echo "  2) Show tweet count"
  echo "  3) Show top liked tweets"
  echo "  4) Show users and their tweet counts"
  echo "  5) Run custom SQL query"
  echo "  6) Show table schema"
  echo "  q) Quit"
  echo ""
  read -p "> " choice
  echo ""
  case $choice in
    1)
      sqlite3 -header -column "$DB_PATH" "SELECT id, username, content, likes, created_at FROM tweets ORDER BY id DESC;"
      ;;
    2)
      sqlite3 "$DB_PATH" "SELECT COUNT(*) as total_tweets FROM tweets;"
      ;;
    3)
      sqlite3 -header -column "$DB_PATH" "SELECT id, username, content, likes FROM tweets ORDER BY likes DESC LIMIT 10;"
      ;;
    4)
      sqlite3 -header -column "$DB_PATH" "SELECT username, COUNT(*) as tweet_count, SUM(likes) as total_likes FROM tweets GROUP BY username ORDER BY tweet_count DESC;"
      ;;
    5)
      read -p "SQL> " query
      sqlite3 -header -column "$DB_PATH" "$query"
      ;;
    6)
      sqlite3 "$DB_PATH" ".schema tweets"
      ;;
    q)
      echo "Bye!"
      exit 0
      ;;
    *)
      echo "Invalid option"
      ;;
  esac
  echo ""
done
