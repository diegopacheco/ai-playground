#!/bin/bash

cd "$(dirname "$0")"

npm run install:all 2>/dev/null

echo "================================================"
echo " SnarkTank"
echo "================================================"
echo ""
echo " Frontend: http://localhost:3000"
echo " Backend:  http://localhost:3001"
echo " Database: SQLite (backend/snarktank.db)"
echo ""
echo " No default user - register a new account."
echo " Registration: username + display name + password (min 6 chars)"
echo ""
echo "================================================"
echo ""

npm run dev
