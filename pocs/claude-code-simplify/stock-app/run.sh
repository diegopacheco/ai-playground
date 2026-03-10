#!/bin/bash
cd "$(dirname "$0")"
mvn clean package -DskipTests -q
java -jar target/stock-app-1.0.0.jar &
echo $! > app.pid
echo "Stock app started on http://localhost:8080 (PID: $(cat app.pid))"
