#!/bin/bash
node bracket-cli.js --sync
node server.js > server.log 2>&1 &
echo $! > .server.pid
echo "Server started on PID $(cat .server.pid). Access the site at: http://localhost:3000/"

